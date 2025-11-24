#pip install ucimlrepo
#pip install scipy

from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
x = car_evaluation.data.features 
y = car_evaluation.data.targets 

# metadata 
#print(car_evaluation.metadata)  
# variable information 
#print(car_evaluation.variables)  

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print('done')