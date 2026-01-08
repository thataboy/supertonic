from supertonic import TTS

# Note: First run downloads model automatically (~260MB)
tts = TTS(auto_download=True)

# Get a voice style
style = tts.get_voice_style(voice_name="M4")

# Generate speech
text = "This morning, I took a walk in the park, and the sound of the birds and the breeze was so pleasant that I stopped for a long time just to listen."
# text = """Je demande pardon aux enfants d'avoir dédié ce livre à une grande personne. J'ai une excuse sérieuse: cette grande personne est le meilleur ami que j'ai au monde. J'ai une autre excuse: cette grande personne peut tout comprendre, même les livres pour enfants"""
wav, duration = tts.synthesize(text, voice_style=style)
# wav: np.ndarray, shape = (1, num_samples)
# duration: np.ndarray, shape = (1,)

# Save to file
tts.save_audio(wav, "results/example_pypi.wav")