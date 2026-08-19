def calculate_scores(audio_results, video_results, nlp_results):
    # --- Safe metric extraction with defaults ---
    
    # Audio
    wpm = audio_results.get("wpm", 0)
    fillers_per_min = audio_results.get("fillers_per_minute", 0.0)
    avg_pause_sec = audio_results.get("avg_pause_sec", 0.0)
    pitch_var = audio_results.get("pitch_variation_hz", 0.0)
    
    # Video
    upright_pct = video_results.get("posture_upright_pct", 0)
    forward_pct = video_results.get("head_pose_forward_pct", 0)
    gestures = video_results.get("gesture_active_pct", 0)
    eye_contact = video_results.get("eye_contact_pct", 0)
    smile = video_results.get("smile_pct", 0)
    
    # NLP
    grammar_errors = nlp_results.get("grammar_errors", 0)
    total_sentences = nlp_results.get("total_sentences", 1)
    if total_sentences == 0: total_sentences = 1
    vocab_ttr = nlp_results.get("vocabulary_ttr", 0.0)

    # --- Calculations ---
    
    # 1. Physical Presence
    physical = (upright_pct * 0.5) + (forward_pct * 0.3) + (gestures * 0.2)
    
    # 2. Vocal Engagement
    if 130 <= wpm <= 160:
        wpm_score = 100
    elif wpm < 130:
        wpm_score = max(0, 100 - (130 - wpm) * 2)
    else:
        wpm_score = max(0, 100 - (wpm - 160) * 2)
        
    pitch_score = min(100, pitch_var * 2) # Assume 50Hz variation is excellent
    vocal = (wpm_score * 0.7) + (pitch_score * 0.3)
    
    # 3. Fluency
    filler_score = max(0, 100 - (fillers_per_min * 10)) # -10 pts per filler/min
    pause_score = 100
    if avg_pause_sec > 2.5:
        pause_score = max(0, 100 - ((avg_pause_sec - 2.5) * 30))
    elif avg_pause_sec < 0.5:
        pause_score = 70 # too robotic/rushed
    fluency = (filler_score * 0.6) + (pause_score * 0.4)
    
    # 4. English Proficiency
    grammar_score = max(0, 100 - ((grammar_errors / total_sentences) * 50))
    vocab_score = min(100, vocab_ttr * 150) # 0.6 TTR -> 90 score
    english = (grammar_score * 0.7) + (vocab_score * 0.3)
    
    # 5. Confidence
    confidence = (eye_contact * 0.4) + (fluency * 0.3) + (physical * 0.3)
    
    # 6. Communication Impact
    impact = (eye_contact * 0.3) + (vocal * 0.3) + (smile * 0.2) + (vocab_score * 0.2)
    
    # 7. Overall Performance
    overall = (physical + vocal + fluency + english + confidence + impact) / 6.0
    
    # Return as a dictionary of rounded values
    return {
        "Physical Presence": round(physical, 1),
        "Vocal Engagement": round(vocal, 1),
        "Fluency": round(fluency, 1),
        "English Proficiency": round(english, 1),
        "Confidence": round(confidence, 1),
        "Communication Impact": round(impact, 1),
        "Overall Performance": round(overall, 1)
    }
