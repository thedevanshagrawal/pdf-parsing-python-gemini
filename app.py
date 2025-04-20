from flask import Flask, request, jsonify
import pdfplumber
from dotenv import load_dotenv
import os
import google.generativeai as genai
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")


# Configure OpenAI client (Gemini API)
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)


@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded."}), 400

    file = request.files["file"]

    if not file.filename.endswith(".pdf"):
        return jsonify({"message": "Invalid file type. Only PDFs are supported."}), 400

    try:
        with pdfplumber.open(file) as pdf:
            resume_text = "\n".join(
                [page.extract_text() for page in pdf.pages if page.extract_text()]
            )

        if not resume_text.strip():
            return jsonify({"message": "PDF appears to be empty or not readable."}), 400

        system_prompt = """
        You are a professional resume analyzer and enhancer. Your task is to analyze the user's resume and, based on the given conditions, provide feedback and suggestions for improvement.
        
        - Identify strengths, weaknesses, and areas for improvement.
        - Provide structured feedback with a summary of the resume and recommendations.
        - Make necessary corrections to improve clarity, readability, and ATS optimization.

        Strengths:
        - Clearly structured sections with relevant details.
        - Includes industry-related keywords (if applicable).
        - Experience and skills align well with the job role.
        - Demonstrates relevant expertise and qualifications.
        
        Weaknesses:
        - Section headings may not follow standard resume format (e.g., "Job History" instead of "Work Experience").
        - Bullet points may be too lengthy, vague, or unclear.
        - Inconsistent formatting (fonts, spacing, alignment).
        - Lacks full ATS (Applicant Tracking System) optimization.

        Recommendations for Improvement:
        - Enhance Section Headings (e.g., "Work Experience", "Education", "Skills").
        - Improve Clarity & Readability (e.g., keep bullet points concise and action-driven, highlight achievements with measurable impact).
        - Optimize for ATS (e.g., use industry keywords, standard date formats, avoid excessive design elements).
        
        Output:
        - Enhanced Resume with ATS-optimized sections and bullet points.
        """

        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": resume_text},
            ],
            stream=True,
        )

        # Collecting the response
        result_data = ""
        for chunk in response:
            result_data += chunk.choices[0].delta.content

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Resume successfully analyzed",
                    "data": result_data,
                }
            ),
            200,
        )

    except Exception as e:
        print("Error:", e)
        return (
            jsonify(
                {"success": False, "message": "Error analyzing resume", "error": str(e)}
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
