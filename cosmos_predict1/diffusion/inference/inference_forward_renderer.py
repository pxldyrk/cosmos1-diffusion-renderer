# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import torch

from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments
from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import envmap_vec
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import process_environment_map
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataset_inference import VideoGBufferDataset
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataloader_utils import dict_collation_fn, dict_collation_fn_concat, sample_continuous_keys

from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.io import save_video, save_image_or_video

torch.enable_grad(False)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add Diffusion Renderer specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Diffusion_Renderer_Forward_Cosmos_7B",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Diffusion_Renderer_Forward_Cosmos_7B",
        ],
    )

    parser.add_argument(
        "--inference_passes",
        type=str,
        default=["rgb"],
        nargs="+",
        help=(
            "List of output passes to generate with the forward renderer. "
            "Typically, only 'rgb' is supported for relighting. "
            "Default: ['rgb']"
        ),
    )
    parser.add_argument(
        "--save_image",
        type=str2bool,
        default=False,
        help=(
            "If True, saves each output frame as an image file in addition to (or instead of) a video. "
            "Default: False."
        ),
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "Path to the input data. Can be a directory containing G-buffer frames, "
            "or a dataset name. This should point to the output of the inverse renderer "
            "(e.g., .../gbuffer_frames/)."
        ),
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default=None,
        nargs="+",
        help=(
            "List of allowed image file extensions to load (e.g., jpg, png, jpeg). "
            "If not set, uses default supported types."
        ),
    )
    parser.add_argument(
        "--resize_resolution",
        type=int,
        default=None,
        nargs="+",
        help=(
            "Resize input images to this resolution before other processing, e.g. center crop. "
            "Provide as two integers: height width. If not set, uses original image size."
        ),
    )

    parser.add_argument(
        "--use_custom_envmap",
        type=str2bool,
        default=True,
        help=(
            "If True, uses user-specified environment maps for relighting (see --envlight_ind). "
            "If False, uses random environment lighting. Default: True."
        ),
    )
    parser.add_argument(
        "--envlight_ind",
        type=int,
        default=[0, 1, 2, 3],
        nargs="+",
        help=(
            "Indices of environment maps to use for relighting. "
            "Each index corresponds to a predefined HDRI in the code. "
            "Ignored if --use_custom_envmap is False. Default: 0 1 2 3"
        ),
    )
    parser.add_argument(
        "--rotate_light",
        type=str2bool,
        default=False,
        help=(
            "If True, rotates the environment lighting for each frame (e.g., for relighting with rotating sun). "
            "Default: False."
        ),
    )
    parser.add_argument(
        "--use_fixed_frame_ind",
        type=str2bool,
        default=False,
        help=(
            "If True, uses a fixed frame index (see --fixed_frame_ind) for lighting rotation, "
            "instead of using the current frame index. "
            "Default: False."
        ),
    )
    parser.add_argument(
        "--fixed_frame_ind",
        type=int,
        default=0,
        help=(
            "Frame index to use for lighting rotation if --use_fixed_frame_ind is True. "
            "Default: 0."
        ),
    )
    return parser.parse_args()


ENV_LIGHT_PATH_LIST = [
    ## default HDRIs
    "asset/examples/hdri_examples/sunny_vondelpark_2k.hdr",
    "asset/examples/hdri_examples/pink_sunrise_2k.hdr",
    "asset/examples/hdri_examples/street_lamp_2k.hdr",
    "asset/examples/hdri_examples/rosendal_plains_1_2k.hdr",
]


def demo(args: argparse.Namespace):
    """Run diffusion renderer inference.
    
    Args:
        args: Command line arguments
    """

    if isinstance(args.envlight_ind, int):
        args.envlight_ind = [args.envlight_ind]
    
    misc.set_random_seed(args.seed)

    # Initialize renderer pipeline
    pipeline = DiffusionRendererPipeline(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_guardrail_models=args.offload_guardrail_models,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
    )

    # Prepare input data
    dataset = VideoGBufferDataset(
        root_dir=args.dataset_path,
        sample_n_frames=args.num_video_frames,
        image_extensions=args.image_extensions,
        resolution=(args.height, args.width),
        resize_resolution=args.resize_resolution,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=dict_collation_fn)

    # Create output directory
    os.makedirs(args.video_save_folder, exist_ok=True)

    # Generate output
    n_test = len(dataloader)
    iter_dataloader = iter(dataloader)
    for i in range(n_test):
        data_batch = next(iter_dataloader)

        if args.use_fixed_frame_ind:
            # use a static frame for g-buffers
            for attributes in ['rgb', 'basecolor', 'normal', 'depth', 'roughness', 'metallic', ]:
                if attributes in data_batch:
                    data_batch[attributes] = data_batch[attributes][:, :, args.fixed_frame_ind:args.fixed_frame_ind + 1, ...].expand_as(data_batch[attributes])

        for envlight_ind in args.envlight_ind:
            # Prepare lighting data
            if args.use_custom_envmap:
                device = torch.device("cuda")
                envlight_path = ENV_LIGHT_PATH_LIST[envlight_ind]
                envlight_dict = process_environment_map(
                    envlight_path,
                    resolution=(args.height, args.width),
                    num_frames=args.num_video_frames,
                    fixed_pose=True,
                    rotate_envlight=args.rotate_light,
                    env_format=['proj', ],
                    device=device,
                )  # Tensors are with shape (T, H, W, 3) in [0, 1]
                env_ldr = envlight_dict['env_ldr'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                env_log = envlight_dict['env_log'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
                env_nrm = envmap_vec([args.height, args.width], device=device)  # [H, W, 3]
                env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)
                data_batch['env_ldr'] = env_ldr
                data_batch['env_log'] = env_log
                data_batch['env_nrm'] = env_nrm
                log.info(f"Using environment map: {envlight_path}")
                current_seed = args.seed
            else:
                # when not using custom envmap, randomize the illumination with different seed
                current_seed = args.seed + int(envlight_ind * 1000)

            # overwrite specified G-buffer passes
            output = pipeline.generate_video(
                data_batch=data_batch,
                seed=current_seed,
            )

            # Save output as individual frames
            if args.save_image:
                video_relative_base_name = data_batch['clip_name'][0]
                for ind in range(output.shape[0]):  # (T, H, W, C)
                    save_path = os.path.join(args.video_save_folder, f"relit_frames_{envlight_ind:04d}", f"{video_relative_base_name}.{ind:04d}.jpg")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_image_or_video(
                        video_save_path=save_path,
                        video=output[ind:ind + 1, ...],
                        H=args.height,
                        W=args.width,
                    )

            # Save output
            clip_name = data_batch['clip_name'][0].replace("/", "__")
            video_save_path = os.path.join(args.video_save_folder, f"{clip_name}.relit_{envlight_ind:04d}.mp4")
            save_image_or_video(
                video_save_path=video_save_path,
                video=output,
                fps=args.fps,
                H=args.height,
                W=args.width,
                video_save_quality=5,
            )
            log.info(f"Saved video to {video_save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    demo(args) 