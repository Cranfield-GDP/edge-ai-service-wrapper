PERMITTED_LANGUAGES = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

def get_permitted_voices(language_code: str) -> list[str]:
    """Get the list of permitted voices for a given language code."""
    voices = {
        "a": [
            "af",
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_heart",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
        ],
        "b": [
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
        ],
        "e": [
            "ef_dora",
            "em_alex",
            "em_santa",
        ],
        "f": [
            "ff_siwis",
            "hf_alpha",
            "hf_beta",
        ],
        "h": [
            "hm_omega",
            "hm_psi",
        ],
        "i": [
            "if_sara",
            "im_nicola",
        ],
        "j": [
            "jf_alpha",
            "jf_gongitsune",
            "jf_nezumi",
            "jf_tebukuro",
            "jm_kumo",
        ],
        "p": [
            "pf_dora",
            "pm_alex",
            "pm_santa",
        ],
        "z": [
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
        ],
    }
    return voices.get(language_code, [])


def prepare_ai_service_request_data():
    """Prepare the `data` part for the AI service request, including text and voice selection."""
    data = {}

    # prompt the user to select a language
    while True:
        print("Available languages:")
        for code, name in PERMITTED_LANGUAGES.items():
            print(f"{code}: {name}")
        language_code = input("Please select a language code: (default to: american english: a) ").strip().lower()
        if language_code in PERMITTED_LANGUAGES:
            data["lang_code"] = language_code
            break
        if not language_code:
            data["lang_code"] = "a"
            break

        print(f"Invalid language code. Please select from {', '.join(PERMITTED_LANGUAGES.keys())}.")
    
    # prompt the user to select a voice
    voices = get_permitted_voices(data["lang_code"])
    if not voices:
        print(f"No voices available for language code '{data['lang_code']}'.")
        return None
    
    while True:
        print("Available voices:")
        for voice in voices:
            print(f"- {voice}")
        voice = input(f"Please select a voice: (default to {voices[0]})").strip()
        if voice in voices:
            data["voice"] = voice
            break
        if not voice:
            data["voice"] = voices[0]
            break

        print(f"Invalid voice. Please select from {', '.join(voices)}.")

    # prompt the user to select a speech speed as a float value
    while True:
        speed = input("Please select a speech speed (default to 1.0): ").strip()
        if not speed:
            data["speed"] = 1.0
            break
        try:
            data["speed"] = float(speed)
            break
        except ValueError:
            print("Invalid speed. Please enter a numeric value.")


    while True:
        # Prompt the user to input the text they want to convert to speech
        data["text"] = input("Please input the text to convert to speech: ")

        if data["text"].strip():
            break

        print("Text cannot be empty. Please try again.")
    
    return data

def prepare_ai_service_request_files():
    """Prepare the `files` part for the AI service request (for FastAPI server)."""
    files = {}
    # Ask the user for any other additional files if needed in future
    return files