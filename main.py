from PIL import Image
from typing import Dict, List, Tuple, Union, Any
import json
from difflib import SequenceMatcher
import cv2
from openai import OpenAI
from anthropic import Anthropic
import numpy as np
from dataclasses import dataclass
import pytesseract
from langchain import OpenAI, PromptTemplate, LLMChain

from constants import OPENAI_KEY, PROMPT_EXTRACTOR, PYTESSERACT_CONFIG

@dataclass
class ProcessedImage:
    image_id: str
    width: str
    height: str

@dataclass
class ExtractionResult:
    micr_number: str 
    confidence_score: str

class InitialImageProcessing:
    @staticmethod
    def processing_image(image: Image.image) -> Tuple[Image.image, np.ndarray]:
        """Preprocess the cheque image before feeding into OCR and LLM"""
        image = cv2.imread(image)
        if image is None:
            raise ValueError("Could not read image from the provided path")
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(thresh)
        processed_image = Image.fromarray(denoised)

        return image, processed_image
    
    @staticmethod
    def resize_image_micr(processed_image: np.ndarray) -> np.ndarray:
        """Extract MICR line from the bottom of the cheque"""
        if processed_image is None:
            raise ValueError("Input of processed image cannot be None")
        height = processed_image.shape[0]
        micr_region = processed_image[int(0.85 * height): height, :]

        return micr_region


class MICRExtraction:
    def __init__(self, openai_key: str):
        """Initialize the cheque extraction pipeline"""
        self.client = OpenAI(OPENAI_KEY)
        self.image_processor = InitialImageProcessing()


    def process_image(self, image_path):
        original_image, processed_image = self.image_processor.processing_image(image_path)
        processed_image_resized = self.image_processor.resize_image_micr(processed_image)

        return processed_image_resized

    def ocr_extractor(self, processed_image_resized: np.ndarray):
        micr_text = pytesseract.image_to_string(processed_image_resized,
                                        config=PYTESSERACT_CONFIG).strip()
        
        ocr_response = {
            "micr_text": micr_text
        }

        return ocr_response
    
    def llm_extractor(self, processed_image_resized: np.ndarray):
        """Extract the MICR using LLM openai"""
        ocr_response = self.ocr_extractor(processed_image_resized)
        prompt = PromptTemplate(input_variables=["ocr_response"], template=PROMPT_EXTRACTOR)
        llm = OpenAI(model="gpt-4o-mini", temperature=0)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        llm_response = llm_chain.run(ocr_response)
        return llm_response


        

