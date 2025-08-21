import json
import os
import numpy as np
import logging
import re
import time
import random
from tqdm import tqdm
from openai import OpenAI
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPT Client - replace with your own API key
client = OpenAI(
    api_key='YOUR_API_KEY_HERE',  # <-- [PLEASE ENTER YOUR API KEY HERE]
    base_url="https://api.gptplus5.com/v1"  # <-- [MODIFY API BASE URL IF NEEDED]
)

def chat_with_gpt(prompt, model_name="gpt-4o"):
    """Use GPT model to process text with exponential backoff retry mechanism"""
    max_retries = 5
    base_delay = 5
    max_delay = 60
    jitter = 0.1

    RETRYABLE_ERRORS = [
        "rate_limit",
        "timeout",
        "connection_error",
        "server_error",
        "500",
        "502",
        "503",
        "504"
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096
            )
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e).lower()
            should_retry = any(err in error_msg for err in RETRYABLE_ERRORS)
            
            if should_retry and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter_amount = delay * jitter
                actual_delay = delay + random.uniform(-jitter_amount, jitter_amount)
                
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries})\n"
                    f"Error: {error_msg}\n"
                    f"Waiting {actual_delay:.2f} seconds before retry..."
                )
                
                time.sleep(actual_delay)
                continue
            else:
                error_type = "Last attempt failed" if attempt == max_retries - 1 else "Non-retryable error"
                logger.error(f"{error_type}: {error_msg}")
                return f"API call failed ({error_type}): {error_msg}"

def parse_continuity_score(result_text):
    """Parse continuity evaluation result"""
    try:
        # Extract score
        score_match = re.search(r"Continuity Score:\s*(\d+(?:\.\d+)?)", result_text)
        explanation_match = re.search(r"Explanation:\s*(.*)", result_text)
        if score_match:
            score = float(score_match.group(1))
            if 0 <= score <= 5:
                return score, explanation_match.group(1) if explanation_match else ""
        logger.warning(f"Could not extract valid score from result: {result_text}")
        return 0, ""
    except Exception as e:
        logger.error(f"Error parsing score: {str(e)}")
        return 0, ""

def evaluate_continuity(predictions, ground_truths):
    """Evaluate continuity of predicted text"""
    system_prompt = """
    You are an expert evaluator specialized in assessing the continuity and coherence of video descriptions.
    Your task is to evaluate how well a series of predicted descriptions maintain continuity compared to the ground truth descriptions.
    
    Please focus on the following aspects:
    1. Temporal Continuity: How well the events flow naturally in time
    2. Action Continuity: How well the actions connect and progress
    3. Subject Consistency: How well the subject's identity and state are maintained
    4. Context Preservation: How well the context and setting are maintained
    5. Logical Flow: How well the descriptions follow a logical sequence
    
    Please provide:
    1. A continuity score from 0 to 5 (where 5 is perfect continuity)
    2. A detailed explanation of your scoring

    
    Format your response as:
    Continuity Score: [score]
    Explanation: [detailed explanation]
    """

    # Build complete prediction and GT text
    pred_text = "\n".join([f"Segment {i+1}: {p}" for i, p in enumerate(predictions)])
    gt_text = "\n".join([f"Segment {i+1}: {g}" for i, g in enumerate(ground_truths)])

    evaluation_prompt = f"""Please evaluate the continuity of the following video descriptions:

    Ground Truth Segments:
    {gt_text}

    Predicted Segments:
    {pred_text}

    Please evaluate how well the predicted segments maintain continuity compared to the ground truth."""

    try:
        result = chat_with_gpt(system_prompt + evaluation_prompt)
        score, explanation = parse_continuity_score(result)
        return {
            "continuity_score": score,
            "explanation": explanation,
            "evaluation_raw_result": result
        }
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return {
            "continuity_score": 0,
            "explanation": "",
            "evaluation_raw_result": f"Evaluation failed: {str(e)}"
        }

def load_data(file_path):
    """
    Load output file data
    
    Args:
        file_path: Path to the output file
    
    Returns:
        List of all video annotations
    """
    all_annotations = []
    
    try:
        # Try to read the entire file as a single JSON object
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'annotations' in data:
                all_annotations.extend(data['annotations'])
    except json.JSONDecodeError:
        # If that fails, try reading line by line as JSONL
        print("Warning: Could not parse file as a single JSON. Trying line by line...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'annotations' in data:
                            all_annotations.extend(data['annotations'])
                    except json.JSONDecodeError:
                        continue
    
    return all_annotations

def calculate_metrics(annotations):
    """
    Calculate Meteor and CIDEr scores
    
    Args:
        annotations: List of annotations containing gt and pred
    
    Returns:
        avg_meteor: Average Meteor score
        avg_cider: Average CIDEr score
    """
    # Initialize scorers
    meteor_scorer = Meteor()
    cider_scorer = Cider()
    
    # Prepare data format
    gts = {}
    res = {}
    
    for i, ann in enumerate(annotations):
        if 'gt' in ann and 'pred' in ann:
            gts[i] = [ann['gt']]
            res[i] = [ann['pred']]
    
    # Calculate scores
    meteor_score, _ = meteor_scorer.compute_score(gts, res)
    cider_score, _ = cider_scorer.compute_score(gts, res)
    
    return meteor_score, cider_score

def calculate_continuity_score(annotations):
    """
    Calculate continuity score
    
    Args:
        annotations: List of annotations containing gt and pred
        
    Returns:
        avg_continuity: Average continuity score
        continuity_details: Continuity score details
    """
    # Organize annotations by video ID
    videos = {}
    for ann in annotations:
        video_id = ann.get('video_id', 'unknown')
        if video_id not in videos:
            videos[video_id] = {'gt': [], 'pred': []}
        videos[video_id]['gt'].append(ann.get('gt', ''))
        videos[video_id]['pred'].append(ann.get('pred', ''))
    
    # Calculate continuity score for each video
    total_score = 0
    continuity_details = {}
    
    print(f"Calculating continuity scores ({len(videos)} videos)...")
    for video_id, content in tqdm(videos.items()):
        # Skip empty videos
        if not content['gt'] or not content['pred']:
            continue
            
        # Evaluate continuity
        eval_result = evaluate_continuity(content['pred'], content['gt'])
        continuity_details[video_id] = eval_result
        total_score += eval_result['continuity_score']
    
    # Calculate average score
    avg_continuity = total_score / len(videos) if videos else 0
    
    return avg_continuity, continuity_details

def evaluate(file_path, output_file=None, eval_continuity=False):
    """
    Evaluate prediction results in the output file
    
    Args:
        file_path: Path to the output file
        output_file: Path to save results (optional)
        eval_continuity: Whether to evaluate continuity (requires OpenAI API)
    """
    # Load data
    print(f"Loading data: {file_path}")
    annotations = load_data(file_path)
    
    # Check if there is enough data for evaluation
    valid_annotations = [ann for ann in annotations if 'gt' in ann and 'pred' in ann]
    if len(valid_annotations) == 0:
        print("No valid annotation data found (requires both gt and pred)")
        return
    
    # Calculate metrics
    print(f"Calculating evaluation metrics ({len(valid_annotations)} samples)...")
    avg_meteor, avg_cider = calculate_metrics(valid_annotations)
    
    # Initialize results
    results = {
        'overall': {
            'meteor': float(avg_meteor),
            'cider': float(avg_cider),
            'sample_count': len(valid_annotations)
        }
    }
    
    # If continuity evaluation is requested
    if eval_continuity:
        print("Starting continuity evaluation...")
        avg_continuity, continuity_details = calculate_continuity_score(valid_annotations)
        results['overall']['continuity'] = float(avg_continuity)
        results['continuity_details'] = continuity_details
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average Meteor score: {avg_meteor:.4f}")
    print(f"Average CIDEr score: {avg_cider:.4f}")
    if eval_continuity:
        print(f"Average Continuity score: {avg_continuity:.4f}")
    
    # Save results to file (if specified)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Meteor, CIDEr, and Continuity scores for predictions")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output", type=str, help="Path to save results (optional)")
    parser.add_argument("--continuity", action="store_true", help="Evaluate continuity using OpenAI API")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (required if --continuity is set)")
    parser.add_argument("--api_base", type=str, default="https://api.gptplus5.com/v1", help="OpenAI API base URL")
    
    args = parser.parse_args()
    
    # Check API key if continuity evaluation is requested
    if args.continuity:
        if not args.api_key:
            print("Error: OpenAI API key is required for continuity evaluation")
            exit(1)
        # Set API key
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base
        )
    
    evaluate(args.input, args.output, args.continuity)