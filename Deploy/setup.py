import os

with open(os.path.join('C:\\Users\\corad\\Desktop\\codenation pp\\AceleraDev-Data-Science-Projeto-Final-2020\\Deploy', 'Procfile'), "w") as file1:
    toFile = 'web: sh setup.sh && streamlit run recommender-aceleradev.py'

    file1.write(toFile)