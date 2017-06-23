# Project organisation
1 . classifier.py: Runs the classifier (default is SVM) on the data set with 80-20 split. Generates a CV accuracy scaore. Generates model is stored in pickle file, classifier.pk

2.  classifier.pk: Trained model generated using the defualt classifier.py

3. veh_det_lib.py : A collection of project specific functions. This library should be imported before running the pipeline.

4. veh_det_pipeline.py: Performs the windowing on an image and applies a pretrained classifier to classifiy windows as cars/no car. The classifier model must have been generated before hand in a file classifier.kk (a defualt is provided)

5. fig_plotter: a utility to plot figures.

