from groq import AsyncGroq
import os
from typing import Tuple
import pandas as pd
from dotenv import load_dotenv
import json
import asyncio
import base64
import logging

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Create async client
client = AsyncGroq(api_key=api_key)

# Configure logger
logger = logging.getLogger(__name__)

def get_valid_best_uses():
    df = pd.read_csv('fabric_restructured.csv')
    # Get unique values and remove any leading/trailing whitespace
    return sorted(list(set(df['Best Use'].str.strip())))

async def classify_requirements(prompt: str) -> Tuple[str, str, str]:
    valid_best_uses = get_valid_best_uses()
    best_uses_str = "\n".join(valid_best_uses)
    
    # Enhanced category mapping with more specific cases
    category_mapping = {
        "leather jacket": "Jackets",
        "biker jacket": "Jackets",
        "summer dress": "Summer-shirts",
        "beach wear": "casual",
        "formal dress": "evening-wear",
        "evening dress": "evening-wear",
        "gown": "evening-gowns",
        "casual dress": "Dresses",
        "party dress": "Party",
        "winter coat": "coats",
        "sports": "Sportswear",
        "workout": "Sportswear",
        "athletic": "Sportswear",
        "everyday": "Everyday",
        "formal": "evening-wear",
        "casual": "casual",
        "summer": "Summer-shirts",
        "winter": "Winter",
        "scarf": "scarves",
        "tie": "ties",
        "blanket": "Blankets",
        "upholstery": "upholstery",
        "curtain": "curtains",
        "bag": "Bags",
        "handbag": "Bags",
        "tent": "tents",
        "active wear": "Activewear",
        "swimsuit": "swimwear",
        "swimming": "swimwear"
    }
    
    try:
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a fabric expert API. Analyze the clothing or textile item request and provide appropriate fabric characteristics.

Your response must be a JSON object with EXACTLY these fields:
{{"best_use": "<use>", "durability": "<durability>", "texture": "<texture>"}}

STRICT REQUIREMENTS:

1. best_use must be ONE of these EXACT values (case-sensitive):
{best_uses_str}

2. durability must be ONE of:
- "High" (for items needing strength, heavy use, outerwear)
- "Moderate" (for regular everyday items)
- "Low" (for delicate, occasional use items)

3. texture must be ONE of:
- "Rough" (for sturdy, textured materials)
- "Smooth" (for flat, sleek materials)
- "Soft" (for comfortable, gentle materials)

GUIDELINES:
- For outerwear (jackets, coats): Usually High durability
- For formal wear: Usually Low/Moderate durability, Smooth/Soft texture
- For sportswear: High durability, Smooth texture
- For everyday wear: Moderate durability
- For luxury items: Often Low durability, Soft texture
- For upholstery: High durability, varies in texture

Example for a leather jacket:
{{"best_use": "Jackets", "durability": "High", "texture": "Smooth"}}

Example for an evening gown:
{{"best_use": "evening-gowns", "durability": "Low", "texture": "Smooth"}}

Respond with ONLY the JSON object, no additional text."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
            temperature=0.3,  # Reduced temperature for more consistent output
            max_tokens=8192,
        )
        
        content = response.choices[0].message.content.strip()
        logger.info(f"Groq API response: {content}")
        
        # Clean up any escaped characters
        content = content.replace('\\"', '"').replace('\\', '')
        
        try:
            result = json.loads(content)
            if not all(k in result for k in ["best_use", "durability", "texture"]):
                raise ValueError("Missing required fields in JSON response")
            
            # Enhanced category mapping logic
            best_use = result["best_use"]
            
            # First check if the input prompt contains any of our mapped categories
            lower_prompt = prompt.lower()
            for key, value in category_mapping.items():
                if key in lower_prompt:
                    best_use = value
                    logger.info(f"Mapped category from prompt '{key}' to '{value}'")
                    break
            
            # If still not in valid categories, try the model's suggestion
            if best_use not in valid_best_uses:
                best_use = category_mapping.get(best_use.lower(), "casual")
                logger.warning(f"Mapped invalid best_use to: {best_use}")
            
            # Final validation
            if best_use not in valid_best_uses:
                logger.warning(f"Invalid best_use after mapping: {best_use}, using 'casual'")
                best_use = "casual"
            
            # Validate durability and texture
            valid_durability = ["High", "Moderate", "Low"]
            valid_texture = ["Rough", "Smooth", "Soft"]
            
            if result["durability"] not in valid_durability:
                result["durability"] = "Moderate"
            if result["texture"] not in valid_texture:
                result["texture"] = "Smooth"
            
            return (best_use, result["durability"], result["texture"])
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {content}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        raise ValueError(f"Error in classification: {str(e)}")

async def classify_requirements_with_image(prompt: str, image_base64: str) -> Tuple[str, str, str, str]:
    valid_best_uses = get_valid_best_uses()
    best_uses_str = "\n".join(valid_best_uses)
    
    # Use the same category mapping as in classify_requirements
    category_mapping = {
        "leather jacket": "Jackets",
        "biker jacket": "Jackets",
        "summer dress": "Summer-shirts",
        "beach wear": "casual",
        "formal dress": "evening-wear",
        "evening dress": "evening-wear",
        "gown": "evening-gowns",
        "casual dress": "Dresses",
        "party dress": "Party",
        "winter coat": "coats",
        "sports": "Sportswear",
        "workout": "Sportswear",
        "athletic": "Sportswear",
        "everyday": "Everyday",
        "formal": "suits",  # Changed from evening-wear to suits for formal attire
        "suit": "suits",
        "casual": "casual",
        "summer": "Summer-shirts",
        "winter": "Winter",
        "scarf": "scarves",
        "tie": "ties",
        "blanket": "Blankets",
        "upholstery": "upholstery",
        "curtain": "curtains",
        "bag": "Bags",
        "handbag": "Bags",
        "tent": "tents",
        "active wear": "Activewear",
        "swimsuit": "swimwear",
        "swimming": "swimwear"
    }
    
    try:
        # First, get a description of the clothing item in the image
        clothing_response = await client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What clothing item or garment is shown in this image? Describe it briefly in one sentence."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.5,
            max_tokens=8192,
            stream=False
        )
        
        clothing_description = clothing_response.choices[0].message.content.strip()
        logger.info(f"Detected clothing item: {clothing_description}")
        
        # Now get the fabric requirements with the clothing description
        response = await client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this image of {clothing_description} and provide fabric requirements.
Return ONLY a JSON object in this exact format:
{{"best_use": "<use>", "durability": "<durability>", "texture": "<texture>"}}

STRICT REQUIREMENTS:

1. best_use must be ONE of these EXACT values (case-sensitive):
{best_uses_str}

2. durability must be ONE of:
- "High" (for items needing strength, heavy use, outerwear)
- "Moderate" (for regular everyday items)
- "Low" (for delicate, occasional use items)

3. texture must be ONE of:
- "Rough" (for sturdy, textured materials)
- "Smooth" (for flat, sleek materials)
- "Soft" (for comfortable, gentle materials)

GUIDELINES:
- For suits and formal wear: Use "suits" category, Moderate durability, Smooth texture
- For outerwear: Usually High durability, category based on specific type
- For everyday wear: Moderate durability, appropriate texture for comfort
- For luxury items: Often Low durability, Soft/Smooth texture"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=512,
            stream=False
        )
        
        content = response.choices[0].message.content.strip()
        logger.info(f"Raw response: {content}")
        
        # Clean up any escaped characters and ensure proper JSON formatting
        content = content.replace('\\"', '"').replace('\\', '')
        if not content.endswith('}'):
            content += '}'
        content = content[content.find('{'):content.rfind('}')+1]
        
        try:
            result = json.loads(content)
            if not all(k in result for k in ["best_use", "durability", "texture"]):
                raise ValueError("Missing required fields in JSON response")
            
            # Apply the same category mapping logic as in classify_requirements
            best_use = result["best_use"]
            
            # Check clothing description for category mapping
            lower_desc = clothing_description.lower()
            for key, value in category_mapping.items():
                if key in lower_desc:
                    best_use = value
                    logger.info(f"Mapped category from description '{key}' to '{value}'")
                    break
            
            # If still not in valid categories, try the model's suggestion
            if best_use not in valid_best_uses:
                best_use = category_mapping.get(best_use.lower(), "casual")
                logger.warning(f"Mapped invalid best_use to: {best_use}")
            
            # Final validation
            if best_use not in valid_best_uses:
                logger.warning(f"Invalid best_use after mapping: {best_use}, using 'casual'")
                best_use = "casual"
            
            # Validate durability and texture
            valid_durability = ["High", "Moderate", "Low"]
            valid_texture = ["Rough", "Smooth", "Soft"]
            
            if result["durability"] not in valid_durability:
                result["durability"] = "Moderate"
            if result["texture"] not in valid_texture:
                result["texture"] = "Smooth"
            
            return (best_use, result["durability"], result["texture"], clothing_description)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Attempted to parse: {content}")
            return ("casual", "Moderate", "Smooth", clothing_description)
            
    except Exception as e:
        logger.error(f"Error in classification with image: {str(e)}")
        raise ValueError(f"Error in classification with image: {str(e)}")

async def get_groq_response(
    original_prompt: str,
    fabric_name: str,
    fabric_type: str,
    best_use: str,
    durability: str,
    texture: str,
    item_description: str = None  # Make it optional for backward compatibility
) -> str:
    try:
        content_text = f"""Original request: {original_prompt}
                    Recommended fabric: {fabric_name}
                    Fabric type: {fabric_type}
                    Best use: {best_use}
                    Durability: {durability}
                    Texture: {texture}"""
        
        if item_description:
            content_text += f"\nDetected item in image: {item_description}"
            
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a fashion consultant API. You must respond with ONLY a valid JSON object.

Format your response exactly like this, with no additional text:
{"recommendation": "your detailed recommendation here"}

The recommendation should be friendly and informative, mentioning the detected clothing item if provided.
The response must be contained within the JSON structure.
Do not include any other text or formatting."""
                },
                {
                    "role": "user",
                    "content": content_text
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up any escaped characters
        content = content.replace('\\"', '"').replace('\\', '')
        
        # Ensure we have valid JSON
        try:
            result = json.loads(content)
            if "recommendation" not in result:
                raise ValueError("Missing 'recommendation' field in JSON response")
            return result["recommendation"]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {content}")  # Debug print
            raise ValueError(f"Invalid JSON response: {str(e)}")
            
    except Exception as e:
        raise ValueError(f"Error in getting recommendation: {str(e)}")

# Add cleanup function
async def cleanup():
    await client.close() 