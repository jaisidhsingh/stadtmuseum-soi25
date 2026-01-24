"""
ComfyUI Workflow Functions

This module handles all ComfyUI/diffusion-related operations:
- Loading and configuring the workflow
- Submitting jobs to ComfyUI server
- Tracking job status via websocket
- Retrieving output images
"""

import json
import uuid
import shutil
import random
import urllib.request
import urllib.parse
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Configuration
COMFYUI_SERVER = "127.0.0.1:8188"
WORKFLOW_API_PATH = Path(__file__).parent / "withoutControlnetAPIF.json"
INPAINT_WORKFLOW_PATH = Path(__file__).parent / "inpaintPersonWorkflow.json"
COMFYUI_INPUT_DIR = Path(__file__).parent / "ComfyUI/input"
COMFYUI_OUTPUT_DIR = Path(__file__).parent / "ComfyUI/output"
COMFYUI_TEMP_DIR = Path(__file__).parent / "ComfyUI/temp"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Node IDs from the workflow (these are stable identifiers)
NODE_LOAD_INPUT_IMAGE = "21"      # LoadImage node for input
NODE_EMPTY_LATENT = "6"           # EmptyLatentImage - controls batch size
NODE_AIO_PREPROCESSOR = "22"      # AIO_Preprocessor - segmentation output
NODE_SAVE_IMAGE = "8"             # SaveImage - final outputs


class JobStatus(str, Enum):
    PROCESSING = "processing"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class JobResult:
    status: JobStatus
    error: Optional[str] = None
    segmented_image: Optional[str] = None
    generated_images: Optional[list[str]] = None


def _load_workflow_api() -> dict:
    """
    Load the API-format workflow JSON.
    This file should be exported from ComfyUI via File -> Export (API).
    """
    with open(WORKFLOW_API_PATH, 'r') as f:
        workflow = json.load(f)
    
    # Remove _meta keys (optional metadata, not needed for execution)
    for node_id in workflow:
        if '_meta' in workflow[node_id]:
            del workflow[node_id]['_meta']
    
    return workflow


def _build_prompt(input_image_filename: str) -> dict:
    """
    Build a prompt with the input image configured.
    """
    import copy
    prompt = copy.deepcopy(_load_workflow_api())
    
    # Configure the input image node
    prompt[NODE_LOAD_INPUT_IMAGE]["inputs"]["image"] = input_image_filename
    
    # Set batch size to 3
    prompt[NODE_EMPTY_LATENT]["inputs"]["batch_size"] = 3
    
    # Randomize seed for variety
    prompt["5"]["inputs"]["seed"] = random.randint(0, 2**63 - 1)
    
    return prompt


def _queue_prompt(prompt: dict, prompt_id: str, client_id: str) -> None:
    """Submit a prompt to the ComfyUI server."""
    p = {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFYUI_SERVER}/prompt", data=data)
    urllib.request.urlopen(req).read()


def _get_history(prompt_id: str) -> dict:
    """Get the execution history for a prompt."""
    try:
        with urllib.request.urlopen(f"http://{COMFYUI_SERVER}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except Exception:
        return {}


def _copy_image_to_outputs(filename: str, subfolder: str, folder_type: str, 
                           output_name: str) -> str:
    """
    Copy an image from ComfyUI's folders to our output directory.
    Returns the relative path for the API response.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if folder_type == "output":
        base_dir = COMFYUI_OUTPUT_DIR
    elif folder_type == "temp":
        base_dir = COMFYUI_TEMP_DIR
    else:
        base_dir = COMFYUI_INPUT_DIR
    
    source = base_dir / subfolder / filename if subfolder else base_dir / filename
    
    dest = OUTPUT_DIR / output_name
    shutil.copy2(source, dest)
    
    return f"/outputs/{output_name}"


def submit_job(input_image_path: str) -> str:
    """
    Submit a new job to ComfyUI.
    
    Args:
        input_image_path: Path to the input image file
        
    Returns:
        job_id: Unique identifier for tracking this job
    """
    job_id = str(uuid.uuid4())
    
    # Copy input image to ComfyUI input directory with unique name
    input_path = Path(input_image_path)
    input_filename = f"soi_input_{job_id}{input_path.suffix}"
    dest_path = COMFYUI_INPUT_DIR / input_filename
    shutil.copy2(input_path, dest_path)
    
    # Build and submit the prompt
    prompt = _build_prompt(input_filename)
    _queue_prompt(prompt, job_id, job_id)
    
    return job_id


def get_job_status(job_id: str) -> JobResult:
    """
    Check the status of a submitted job.
    
    Args:
        job_id: The job identifier returned by submit_job
        
    Returns:
        JobResult with status and outputs if finished
    """
    history = _get_history(job_id)
    
    if not history or job_id not in history:
        return JobResult(status=JobStatus.PROCESSING)
    
    job_history = history[job_id]
    
    # Check for errors
    if job_history.get("status", {}).get("status_str") == "error":
        error_msg = job_history.get("status", {}).get("messages", [])
        return JobResult(
            status=JobStatus.FAILED, 
            error=str(error_msg) if error_msg else "Unknown error"
        )
    
    # Check if outputs are available
    outputs = job_history.get("outputs", {})
    
    if not outputs:
        return JobResult(status=JobStatus.PROCESSING)
    
    # Extract outputs
    segmented_image = None
    generated_images = []
    
    # Get segmented image from AIO_Preprocessor (via PreviewImage node 23)
    # Actually, PreviewImage doesn't save - we need to check what's in outputs
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for i, img in enumerate(node_output["images"]):
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                folder_type = img.get("type", "output")
                
                # Determine if this is the preview (temp) or final output
                if folder_type == "temp":
                    # This is the segmented preview
                    if not segmented_image:
                        output_name = f"seg_{job_id}.png"
                        segmented_image = _copy_image_to_outputs(
                            filename, subfolder, folder_type, output_name
                        )
                else:
                    # These are the generated images
                    output_name = f"gen_{job_id}_{i}.png"
                    generated_images.append(
                        _copy_image_to_outputs(filename, subfolder, folder_type, output_name)
                    )
    
    if generated_images:
        return JobResult(
            status=JobStatus.FINISHED,
            segmented_image=segmented_image,
            generated_images=generated_images
        )
    
    return JobResult(status=JobStatus.PROCESSING)



def wait_for_job(job_id: str, timeout: float = 60.0) -> JobResult:
    """
    Synchronously wait for a job to complete.
    Polls every 0.5 seconds.
    """
    import time
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        result = get_job_status(job_id)
        if result.status in [JobStatus.FINISHED, JobStatus.FAILED]:
            return result
        time.sleep(0.5)
    
    return JobResult(status=JobStatus.PROCESSING, error="Timeout waiting for job")

