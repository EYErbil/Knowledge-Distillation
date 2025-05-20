# Knowledge-Distillation- Queen's University Elec475-TermProject competition winner.
Applied knowledge distillation technique for machine learning models, as source code.
More info --> https://neptune.ai/blog/knowledge-distillation
The results of the distillation can be seen in results excel file, with all necessary information available on either the code/predictions-visualizations.
# For feature distillation technique, backbone has to be similar
## To train in both ways:
python train_kd.py --kd_method response --model_name light --temperature 4 --alpha 0.3 
python train_kd.py --kd_method feature --model_name light
python train.py --model_name light
## to test
python test.py --training_mode teacher
python test.py --training_mode regular
python test.py --training_mode response
python test.py --training_mode feature
## to see the model (prepared by me, excels in feature extraction)
python model.py --m light (for model + torchsummary)

