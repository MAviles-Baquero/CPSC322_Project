# CPSC322_Project
Spring of 2021 Course Project for Gina Sprint

Our Project:
In our project we utilized the Household Pulse Survey to explore the effect of various socioeconomic factors on delayed or canceled health care during the COVID-19 pandemic. 

To Run:
- Go to https://www.census.gov/programs-surveys/household-pulse-survey/datasets.html
- Select and download 'HPS Week 21 PUF SAS'
- Extract the data inside it to input_data
- Run the eda.iypnb to create week21_working.csv and then run Project_Report.ipynb
- type in the below string, while trying out various different values, to see our api in action
- https://health-delay-predictor.herokuapp.com/predict?birth_year=1935&gender=1&hispanic=2&race=4&income=3&education=7

Organization:
- Project_Proposal.ipynb was the original proposal of the project
- Project_Report.ipynb is the main file of analysis, featuring our EDA and classifier analysis
- api.py, heroku.yml, forest_pickler.py, and dockerfile were all used to deploy our api. In master they are configured for a localhost, but in our api branch they are configured in the format that was needed to deploy them on heroku.
- input_data contained all our input data and processed input data
- mysklearn was our own classifier, utility, and all our other functions that we felt the need to define
- test_myforest.py was our test file for our forest
