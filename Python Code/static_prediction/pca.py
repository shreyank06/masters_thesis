import numpy as np
import tensorflow as tf
#https://towardsdatascience.com/all-you-need-to-know-about-pca-technique-in-machine-learning-443b0c2be9a1

class PCA:
    def __init__(self, df, n_components):
        self.df = df
        self.n_components = n_components

    def fit_transform(self):
        train_tensor = tf.constant(self.df.values, dtype=tf.float32)

        # Compute covariance matrix
        cov_matrix = tf.matmul(tf.transpose(train_tensor), train_tensor)

        # Ensure symmetric covariance matrix
        cov_matrix = 0.5 * (cov_matrix + tf.transpose(cov_matrix))

        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = tf.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        eigenvalues_indices = tf.argsort(tf.math.real(eigenvalues), direction='DESCENDING')
        sorted_eigenvalues = tf.gather(eigenvalues, eigenvalues_indices)
        sorted_eigenvectors = tf.gather(eigenvectors, eigenvalues_indices, axis=1)

        # Select top k eigenvectors corresponding to the top k eigenvalues
        pca_matrix = tf.slice(sorted_eigenvectors, [0, 0], [train_tensor.shape[1], self.n_components])

        with tf.compat.v1.Session() as sess:
            pca_matrix_val = sess.run(pca_matrix)
            total_variance = sess.run(tf.reduce_sum(sorted_eigenvalues))
            variance_ratio = sess.run(sorted_eigenvalues / total_variance)

            print("Variance captured by each principal component:")
            for i in range(self.n_components):
                print(f"Principal Component {i+1}: {variance_ratio[i]}")

        # Transform the data
        transformed_data = np.dot(self.df.values, pca_matrix_val)

        return transformed_data

    
    # def print_mapping():
    #     # Print the mapping of original columns to principal components
    #     print("Mapping of Original Columns to Principal Components:")
    #     original_column_names = [column for column, index in column_indices.items()]
    #     for i in range(pca_matrix_val.shape[1]):
    #         principal_component = pca_matrix_val[:, i]
    #         column_contributions = [(original_column_names[j], principal_component[j]) for j in range(len(original_column_names))]
    #         sorted_contributions = sorted(column_contributions, key=lambda x: abs(x[1]), reverse=True)
    #         print(f"Principal Component {i+1}:")
    #         for column_name, contribution in sorted_contributions:
    #             print(f"\t{column_name}: {contribution}")
