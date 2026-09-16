import os
import sys
import json
import base64
import argparse
import requests

def encode_image_base64(image_path):
    """Encode local image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Test VLM Client for Video Visual Relation Detection (VidVRD)")
    parser.add_argument(
        "--payload",
        type=str,
        default="vlm_prompt_payload.json",
        help="Path to VLM prompt payload JSON file"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://10.x.x.x:8000/v1/chat/completions",
        help="URL of served Qwen2-VL-2B model endpoint"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/simulation mode without connecting to live server"
    )
    return parser.parse_args()

def get_mock_triplets(payload_name):
    if "abandoned" in payload_name.lower():
        return [
            {"subject": "[1]", "relation": "touch", "object": "[2]"},
            {"subject": "[2]", "relation": "hug", "object": "[1]"},
            {"subject": "[1]", "relation": "get_off", "object": "[4]"}
        ]
    else:
        return [
            {"subject": "[1]", "relation": "wave", "object": "[2]"},
            {"subject": "[1]", "relation": "hold", "object": "[16]"},
            {"subject": "[1]", "relation": "carry", "object": "[16]"}
        ]

def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    payload_file = os.path.join(base_dir, args.payload) if not os.path.isabs(args.payload) else args.payload

    if not os.path.exists(payload_file):
        print(f"[ERROR] Payload file not found: {payload_file}")
        sys.exit(1)

    print("=" * 60)
    print("COMPUTER VISION CENTER - VLM INFERENCE CLIENT")
    print("=" * 60)
    print(f"Loading payload from: {payload_file}")

    with open(payload_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    task_name = payload.get("task", "VidVRD Task")
    target_model = payload.get("model_target", "Qwen2-VL-2B-Instruct")
    frames_seq = payload.get("visual_prompt_frames_sequence", [])
    allowed_relations = payload.get("allowed_relations_vocabulary_26", [])
    system_prompt = payload.get("vlm_system_prompt", "")
    user_prompt = payload.get("vlm_user_prompt", "")

    # Reinforce vocabulary in system prompt to prevent VLM hallucination
    vocab_str = ", ".join(allowed_relations)
    if vocab_str not in system_prompt:
        system_prompt += f"\n\nSTRICT ALLOWED 26 RELATIONS VOCABULARY:\n[{vocab_str}]"

    # Determine frames directory dynamically
    if "tuned" in args.payload.lower():
        frames_dir = os.path.join(base_dir, "data", "vlm_input_frames_abandoned_tuned")
    elif "abandoned" in args.payload.lower():
        frames_dir = os.path.join(base_dir, "data", "vlm_input_frames_abandoned")
    else:
        frames_dir = os.path.join(base_dir, "data", "vlm_input_frames")

    print(f"Task: {task_name}")
    print(f"Target Model: {target_model}")
    print(f"Input Frames: {len(frames_seq)} frames from {frames_dir}")
    print(f"Allowed Relations Vocabulary: {len(allowed_relations)} categories")
    print(f"Target Server Endpoint: {args.server_url}")
    print("-" * 60)

    # Check images exist
    verified_images = []
    for fn in frames_seq:
        fp = os.path.join(frames_dir, fn)
        if os.path.exists(fp):
            verified_images.append(fp)
        else:
            print(f"[WARN] Frame not found: {fp}")

    print(f"Verified {len(verified_images)}/{len(frames_seq)} visual prompt frames ready.")

    # Check mode: mock vs live
    is_live = False
    if not args.mock and "10.x.x.x" not in args.server_url:
        is_live = True

    if is_live:
        print(f"\n[LIVE MODE] Connecting to model server: {args.server_url} ...")
        content_parts = []
        for img_path in verified_images:
            b64_data = encode_image_base64(img_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_data}"
                }
            })
        content_parts.append({
            "type": "text",
            "text": user_prompt
        })

        request_data = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts}
            ],
            "temperature": 0.0,
            "max_tokens": 512
        }

        try:
            resp = requests.post(args.server_url, json=request_data, timeout=30)
            resp.raise_for_status()
            result_json = resp.json()
            raw_text = result_json["choices"][0]["message"]["content"]
            print("[SUCCESS] Received response from model server:")
            print(raw_text)
            
            # Robust extraction of raw JSON array in case VLM wraps in markdown code block
            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            triplets = json.loads(clean_text)
        except Exception as e:
            print(f"[ERROR] Connection to server failed: {e}")
            print("[FALLBACK] Switching to Mock verification to validate pipeline parsing...")
            triplets = get_mock_triplets(args.payload)
    else:
        print("\n[MOCK MODE] Simulating Qwen2-VL-2B inference on Set-of-Marks prompt...")
        print("(Pass --server-url <IP:PORT> to connect to real server)")
        triplets = get_mock_triplets(args.payload)

    # Validate output triplets
    print("\n" + "=" * 60)
    print("INFERENCE RESULT: PREDICTED VISUAL RELATION TRIPLETS")
    print("=" * 60)
    print(f"{'Subject':<12} | {'Relation':<18} | {'Object':<12} | {'Vocabulary Valid'}")
    print("-" * 60)

    all_valid = True
    for t in triplets:
        sub = t.get("subject", "")
        rel = t.get("relation", "")
        obj = t.get("object", "")
        is_rel_valid = rel in allowed_relations
        if not is_rel_valid:
            all_valid = False
        valid_mark = "[OK] Valid" if is_rel_valid else "[X] Invalid"
        print(f"{sub:<12} | {rel:<18} | {obj:<12} | {valid_mark}")

    print("-" * 60)
    if all_valid:
        print("[CHECK PASSED] All predicted relations strictly adhere to standard 26 taxonomy!")
    else:
        print("[CHECK WARNING] Some relations fall outside the 26 allowed vocabulary.")

if __name__ == "__main__":
    main()
