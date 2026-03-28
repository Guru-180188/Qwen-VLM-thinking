import torch
import cv2
from utils import fetch_frame
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import tempfile
import os

# Using the Model ID so transformers automatically finds your .cache folder
model_id = "Qwen/Qwen3-VL-4B-Thinking-FP8"

# 1. Load Model and Processor
# We use trust_remote_code=True for the latest Qwen3 logic
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto", 
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id)

# Process 100 frames
for send_count in range(100):
    #frame from camera
    frame = fetch_frame()
    if frame is None:
        print(f"Error: Could not capture frame at send {send_count + 1}")
        break

    # Save frame temporarily to a file
    temp_dir = tempfile.gettempdir()
    frame_path = os.path.join(temp_dir, f"frame_{send_count}.jpg")
    cv2.imwrite(frame_path, frame)

    # 2. Prepare Multimodal Input
    # Use the saved frame file path
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": frame_path
                },
                {
                    "type": "text", 
                    "text": "Analyze this image. First, show your internal reasoning, then give a final description."
                },
            ],
        }
    ]

    # 3. Processing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # 4. Generate (Thinking models benefit from higher max_new_tokens for the 'thought' block)
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"Send {send_count + 1}/100: {output_text[0]}")
    
    # Clean up temporary frame file
    try:
        os.remove(frame_path)
    except:
        pass