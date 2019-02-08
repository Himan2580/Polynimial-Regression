# Polynimial-Regression
this is the Code for polynomial Regression in Machine learning on simple Data for only understanding.
dataset
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:,2].values

#splitting the dataset into the training set and test set
"""
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state = 0)"""

##Feature Scaling
""""
sc_X = StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)"""

###fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
###fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree= 4) #converting  the featyres into polynomial
X_poly=poly_reg.fit_transform(X)###fitting the X features and transforming it to X-poly
lin_reg_2=LinearRegression() ### making a linear regression model of X_poly
lin_reg_2.fit(X_poly,y)
###Visualizing the linear regression result
plt.scatter(X, y,color="red")
plt.plot(X, lin_reg.predict(X) , color="blue")
plt.title("Truth or bluff(linearRegression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
###Visualizing the polynomial regression results
X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y,color="red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("Truth or Bluff (ploynomialRegression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

##predicting the new result with linear  regression
lin_reg.predict(6.5)
###predicting the new result with polynomial regression
 lin_reg_2.predict(poly_reg.fit_transform(6.5))
