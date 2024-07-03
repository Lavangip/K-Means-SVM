from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn import datasets
iris = datasets.load_iris ( as_frame=True )

#Selecting only 'setosa' and 'versicolor' classes
selected_classes = iris.target.isin([0, 1])
iris_binary = iris.data[selected_classes]
target_binary = iris.target[selected_classes]

#Extracting 'petal length' and 'petal width' features
X = iris_binary[['petal length (cm)', 'petal width (cm)']]
y = target_binary

#Normalizing the dataset
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

#Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)



#Training a Linear Support Vector Classifier
linear_svc = LinearSVC(random_state=42)
linear_svc.fit(X_train, y_train)

print("LinearSVC trained successfully.")

def plot_decision_boundary_save(clf, X, y, title, save_path):
    plt.figure(figsize=(8, 6))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot training data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title(title)


    plt.savefig(save_path)
    plt.close()

plot_decision_boundary_save(linear_svc, X_train, y_train, "Decision Boundary on Training Data", "/content/training_decision_boundary.png")

# Saving scatterplot of test data along with original decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Scatterplot of Test Data with Original Decision Boundary')

# Retrieving the coefficients and intercept from the trained LinearSVC
coef = linear_svc.coef_[0]
intercept = linear_svc.intercept_

# Ploting the decision boundary
x_vals = np.arange(-2, 2, 0.1)
y_vals = -(coef[0] / coef[1]) * x_vals - intercept / coef[1]
plt.plot(x_vals, y_vals, color='black')

plt.savefig("/content/test_scatterplot_with_decision_boundary.png")
plt.close()


# Generating synthetic dataset
X_synthetic, y_synthetic = make_moons(n_samples=500, noise=0.05, random_state=42)

# Adding 5% noise to the dataset
num_noise_points = int(0.05 * len(X_synthetic))
random_indices = np.random.choice(len(X_synthetic), num_noise_points, replace=False)
y_synthetic[random_indices] = 1 - y_synthetic[random_indices]  # Flipping labels

print("Shape of synthetic dataset:", X_synthetic.shape)
print("Number of misclassifications:", np.sum(y_synthetic != (1 - y_synthetic)))


# Defining SVM models with different kernels
svm_linear = SVC(kernel='linear', random_state=42)
svm_poly = SVC(kernel='poly', degree=3, gamma='auto', random_state=42)  # Polynomial kernel with degree 3
svm_rbf = SVC(kernel='rbf', gamma='auto', random_state=42)  # RBF kernel

# Fiting SVM models to the synthetic dataset
svm_linear.fit(X_synthetic, y_synthetic)
svm_poly.fit(X_synthetic, y_synthetic)
svm_rbf.fit(X_synthetic, y_synthetic)


def plot_decision_boundary_save(model, X, y, title, save_path):
    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


plot_decision_boundary_save(svm_linear, X_synthetic, y_synthetic, 'Linear Kernel', '/content/svm_linear_decision_boundary.png')
plot_decision_boundary_save(svm_poly, X_synthetic, y_synthetic, 'Polynomial Kernel', '/content/svm_poly_decision_boundary.png')
plot_decision_boundary_save(svm_rbf, X_synthetic, y_synthetic, 'RBF Kernel', '/content/svm_rbf_decision_boundary.png')



# Defining the parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1]}

svm_rbf_tuned = SVC(kernel='rbf', random_state=42)

grid_search = GridSearchCV(estimator=svm_rbf_tuned, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Performing grid search
grid_search.fit(X_synthetic, y_synthetic)

print("Best hyperparameters:", grid_search.best_params_)

# Get the best SVM model
best_svm_rbf = grid_search.best_estimator_


def plot_decision_boundary_save(model, X, y, title, save_path):
    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# Plot decision boundary with best hyperparameters and save the plot
plot_decision_boundary_save(best_svm_rbf, X_synthetic, y_synthetic, 'RBF Kernel SVM with Best Hyperparameters', '/content/svm_rbf_decision_boundary_best_hyperparameters.png')
