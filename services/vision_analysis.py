import base64
import json
import logging
from typing import cast, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionContentPartParam

logger = logging.getLogger("vision_analysis")

class VisionAnalysisService:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def analyze_student_work(
        self,
        images: Union[list[bytes], bytes],
        lesson_context: dict,
        current_problem: str
    ) -> dict:
        # Normalize to list of bytes
        if isinstance(images, bytes):
            images = [images]

        # Construct prompt text
        text_prompt = f"""
You are looking at one or more images containing the student's handwritten math solution.
Please verify the student's work step-by-step.

### Lesson Context
- Topic: {lesson_context.get('topic', 'N/A')}
- Current Problem: {current_problem}
- Grade Level: {lesson_context.get('grade_level', 'N/A')}
- Conversation History:
\"\"\"
{lesson_context.get('conversation_history', '')}
\"\"\"

### Instructions for verification:
1. Transcribe each line written by the student in the image(s) literally. Do NOT correct any errors, signs, or numbers in your transcription. Look only at what is actually written on the paper.
2. Independently solve the original equation step-by-step in your own scratchpad.
3. Compare the student's literally transcribed steps to the correct mathematical steps. Look out for:
   - Sign errors: did they move a term across the '=' sign but forgot to change its sign (e.g. changing -4 to -4 or +5 to +5)?
   - Arithmetic errors: check every addition, subtraction, multiplication, and division.
   - Final value of x: does it actually solve the original equation?
4. Do NOT hallucinate correct steps that are not in the student's handwritten image(s). Look only at what is actually written on the paper.

### Expected Output Format
You MUST return a JSON object with the following fields in this exact order:
- "transcription": array of strings (the literal, step-by-step transcription of each line written in the student's image, e.g. ["3x - 4 = 11", "3x = 11 - 4", "3x = 12", ...])
- "correct_solution_scratchpad": string (write out the correct mathematical step-by-step solution to the original equation)
- "comparison_scratchpad": string (compare each transcribed student line to the mathematically correct steps. Point out any errors in signs, arithmetic, or logic for each line.)
- "correct": boolean (true if the student's final answer and ALL intermediate steps are correct, false otherwise)
- "confidence": number (between 0.0 and 1.0 representing your confidence in the analysis)
- "feedback": string (clear, step-by-step constructive explanation of what the student did right or wrong, referencing the specific lines in their work)
- "identified_mistakes": array of strings (specific errors in calculations, signs, or reasoning)
- "next_hint": string or null (the next guiding hint/question to lead the student towards the correct solution, without giving away the final answer immediately if they are wrong)
"""

        system_prompt = (
            "You are a highly rigorous expert mathematics tutor.\n\n"
            "Analyze the student's handwritten work shown in the image(s).\n\n"
            "Do NOT assume the student's solution is correct just because it matches the final answer or expected shape. Check every single written line step-by-step for mathematical validity.\n\n"
            "First literally transcribe what is written in the image(s) line-by-step. Do not correct the student's errors in your transcription.\n"
            "Then determine:\n"
            "1. Is the final answer correct?\n"
            "2. Are intermediate steps correct? (Check every line carefully!)\n"
            "3. Where does reasoning break down? (Identify the exact line number/step where the error occurred)\n"
            "4. What misconception is present?\n"
            "5. What hint should be given?\n\n"
            "Return structured JSON only."
        )

        try:
            print("\n" + "="*50)
            print("--- GPT VISION EVALUATION REQUEST ---")
            print("="*50 + "\n")

            user_content: list[ChatCompletionContentPartParam] = [{"type": "text", "text": text_prompt}]
            for img in images:
                base64_image = base64.b64encode(img).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

            logger.info("Sending request to GPT Vision model...")
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=800,
                temperature=0.2,
            )

            result_str = response.choices[0].message.content
            if not result_str:
                raise ValueError("GPT Vision response content is empty or None")

            print("\n" + "="*50)
            print("--- GPT VISION RESPONSE ---")
            print(result_str)
            print("="*50 + "\n")

            logger.info(f"Received vision analysis response: {result_str}")

            result = json.loads(result_str)

            # Ensure all required keys exist with default fallbacks
            if "transcription" not in result:
                result["transcription"] = []
            if "correct_solution_scratchpad" not in result:
                result["correct_solution_scratchpad"] = ""
            if "comparison_scratchpad" not in result:
                result["comparison_scratchpad"] = ""
            if "correct" not in result:
                result["correct"] = False
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "feedback" not in result:
                result["feedback"] = "Could not analyze the work clearly."
            if "identified_mistakes" not in result:
                result["identified_mistakes"] = []
            if "next_hint" not in result:
                result["next_hint"] = None

            return cast(dict, result)

        except Exception as e:
            logger.error(f"Error calling GPT Vision API: {e}")
            raise e
