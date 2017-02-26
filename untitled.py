import matplotlab as plt
import numpy as np

def plot_learning_curve(estimator, title, X, Y, train_sizes, cv=None)
	plt.figure()
	plt.title(title)
	plt.xlabel("Train sets size")
	plt.ylabel("Scores")
	train_sizes, train_scores, cv_scores = learning_curve(estimator=estimator, X=X, Y=Y, cv=cv, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)
	plt.grid()
	plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.1, color="r")
	plt.fill_between(train_sizes, cv_scores_mean-cv_scores_std, cv_scores_mean+cv_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Train score')
	plt.plot(train_sizes, cv_scores_mean, 'o-', color='g', label="CV score")
	plt.legend("best")
	return plt