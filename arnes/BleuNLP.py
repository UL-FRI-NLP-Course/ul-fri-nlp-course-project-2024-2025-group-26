from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# your long ground‑truth and your generated output
gt = ("Podatki o prometu. Zaradi pokvarjenega vozila je na štajerski avtocesti v predoru Jasovnik zaprt vozni pas proti Ljubljani. Na mejnih prehodih Sečovlje, Petrina, Dragonja in Dobovec vozniki osebnih vozil na izstop iz države čakajo približno pol ure; na Obrežju, v Gruškovju ter Slovenski vasi pa 1 uro pri vstopu v državo.")
hyp = "Podatki o prometu. Zaradi prometne nesreče je zaprta regionalna cesta Ajševica-Rožna Dolina, in to pri Ajševici. Na mejnem prehodu Obrežje vozniki na vstop v državo čakajo do dve uri, v Gruškovju pa pol ure. Povečan promet pri izstopu iz države pa je na prehodu Dobovec, na katerem vozniki čakajo uro in pol, ter na Obrežju in v Gruškovju, v katerem vozniki čakajo pol ure."

# tokenize (for Slovenian you may need a custom tokenizer; for quick tests whitespace split also works)
ref_tokens = word_tokenize(gt, language='slovene')
hyp_tokens = word_tokenize(hyp, language='slovene')

# compute sentence‑level BLEU with smoothing
smooth = SmoothingFunction().method1
bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
print(f"BLEU score: {bleu:.4f}")