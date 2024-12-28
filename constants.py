import os


OPENAI_KEY = os.getenv("OPENAI_KEY")
PYTESSERACT_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:⑆"
PROMPT_EXTRACTOR = """
         Your task is to act as MICR extractor. Extract the MICR value and return it in the JSON format.
         - micr text (str)

         MICR Text:
        {ocr_response}
        """