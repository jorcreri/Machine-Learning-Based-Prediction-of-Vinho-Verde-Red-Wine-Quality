datos <- read.csv("winequality-red.data", sep = ";")

str(datos)

X <- datos[, 1:11]
y <- datos$quality

X_scaled <- scale(X)

pca_result <- prcomp(X_scaled, center = TRUE, scale. = TRUE)
summary(pca_result)

grDevices::windows()
plot(pca_result, type = "l", main = "Scree Plot")

loadings <- abs(pca_result$rotation[, 1:6])
variable_influence <- rowSums(loadings)
top_variables <- sort(variable_influence, decreasing = TRUE)
selected_variables <- names(top_variables)[1:6]
selected_variables

loadings <- pca_result$rotation

round(loadings[, 1:6], 3)

selected_data <- datos[, c(2, 4, 5, 8, 10, 11)]

cor_matrix <- cor(selected_data)

print("Matriz de correlación:")
print(cor_matrix)

iqr_values <- apply(selected_data, 2, IQR)

print("Rangos intercuartílicos (IQR):")
print(iqr_values)

precisiones <- c(0.676, 0.719, 0.694, 0.701, 0.721, 0.704, 0.653, 0.677)
grDevices::windows()
barplot(precisiones,
        ylim = c(0, 1),
        col = "steelblue",
        ylab = "Precisión",
        xlab = "",
        main = "Precisión por ejecución del modelo RNA",
        names.arg = rep("", length(precisiones)))  # Sin etiquetas en el eje x

abline(h = seq(0, 1, 0.1), col = "gray90", lty = "dotted")

precisiones <- c(0.721, 0.721, 0.759, 0.767, 0.734, 0.732, 0.715, 0.674)
grDevices::windows()
barplot(precisiones,
        ylim = c(0, 1),
        col = "steelblue",
        ylab = "Precisión",
        xlab = "",
        main = "Precisión por ejecución del modelo SVC",
        names.arg = rep("", length(precisiones)))  # Sin etiquetas en el eje x

abline(h = seq(0, 1, 0.1), col = "gray90", lty = "dotted")

precisiones <- c(0.742, 0.742, 0.730, 0.736, 0.736, 0.739, 0.738, 0.742)
grDevices::windows()
barplot(precisiones,
        ylim = c(0, 1),
        col = "steelblue",
        ylab = "Precisión",
        xlab = "",
        main = "Precisión por ejecución del modelo DoME",
        names.arg = rep("", length(precisiones)))  # Sin etiquetas en el eje x

abline(h = seq(0, 1, 0.1), col = "gray90", lty = "dotted")

precisiones <- c(0.756, 0.752, 0.754, 0.759, 0.758, 0.762)
grDevices::windows()
barplot(precisiones,
        ylim = c(0, 1),
        col = "steelblue",
        ylab = "Precisión",
        xlab = "",
        main = "Precisión por ejecución del modelo arboles de decisiones",
        names.arg = rep("", length(precisiones)))  # Sin etiquetas en el eje x

abline(h = seq(0, 1, 0.1), col = "gray90", lty = "dotted")

precisiones <- c(0.741, 0.742, 0.737, 0.742, 0.737, 0.738)
grDevices::windows()
barplot(precisiones,
        ylim = c(0, 1),
        col = "steelblue",
        ylab = "Precisión",
        xlab = "",
        main = "Precisión por ejecución del modelo kNN",
        names.arg = rep("", length(precisiones)))  # Sin etiquetas en el eje x

abline(h = seq(0, 1, 0.1), col = "gray90", lty = "dotted")

matriz <- matrix(c(3106, 1080, 1169, 2640), nrow = 2, byrow = FALSE)
print(matriz)

matriz <- matrix(c(662, 193, 180, 594), nrow = 2, byrow = FALSE)
print(matriz)

matriz <- matrix(c(632, 232, 180, 564), nrow = 2, byrow = FALSE)
print(matriz)

matriz <- matrix(c(667, 188, 192, 552), nrow = 2, byrow = FALSE)
print(matriz)

matriz <- matrix(c(651, 204, 209, 535), nrow = 2, byrow = FALSE)
print(matriz)
