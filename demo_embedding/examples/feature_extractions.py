from transformers import pipeline

checkpoint = "facebook/bart-base"
feature_extractor = pipeline("feature-extraction", framework="pt", model=checkpoint)
text = "Transformers is an awesome library!"

#Reducing along the first dimension to get a 768 dimensional array
print(feature_extractor(text,return_tensors = "pt")[0].numpy().mean(axis=0))