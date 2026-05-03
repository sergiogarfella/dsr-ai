library(readr)
library(tidyverse)
library(GGally)
library(car)
library(lmtest)

dua <- read_csv("/home/acust/Proyectos/dsr-playground/dataset_unificado_analisis.csv")

# Analisis multivariante
library(tidyverse)
library(patchwork)
amd = dua %>% filter(dataset== "aclImdb")
stan = dua %>% filter(dataset== "stanfordSentimentTreebank")
review = dua %>% filter(dataset== "review_polarity")

#amd
amd_densi <- ggplot(data =amd, aes(x=longitud, fill=sentimiento)) + geom_density(alpha = 0.35) + scale_fill_manual(values = c("pos"="#2E86AB", "neg"="#A23B72"))+ scale_color_manual(values = c("pos"="#2E86AB", "neg"="#A23B72")) + labs(title = "AclImdb", x= "Longitud", y="Sentimiento", fill="Grupo") + theme_minimal() 
amd_box <- ggplot(data = amd, aes(x=sentimiento, y= longitud, fill=sentimiento)) + geom_boxplot(fill= c("#A23B72","#2E86AB")) + labs(title = "AclImdb", x= "Sentimiento", y="Longitud") + theme_minimal() + theme(legend.position = "none")
#stan
stan_densi <- ggplot(data =stan, aes(x=longitud, fill=sentimiento)) + geom_density(alpha = 0.35) + scale_fill_manual(values = c("pos"="#2E86AB", "neg"="#A23B72"))+ scale_color_manual(values = c("pos"="#2E86AB", "neg"="#A23B72")) + labs(title = "StanfordSentimentTreebank", x= "Longitud", y="Sentimiento", fill="Grupo") + theme_minimal() 
stan_box <-ggplot(data = stan, aes(x=sentimiento, y= longitud, fill=sentimiento)) + geom_boxplot(fill= c("#A23B72","#2E86AB")) + labs(title = "StanfordSentimentTreebank", x= "Sentimiento", y="Longitud") + theme_minimal() + theme(legend.position = "none")
#review
review_densi <- ggplot(data =review, aes(x=longitud, fill=sentimiento)) + geom_density(alpha = 0.35) + scale_fill_manual(values = c("pos"="#2E86AB", "neg"="#A23B72"))+ scale_color_manual(values = c("pos"="#2E86AB", "neg"="#A23B72")) + labs(title = "Review_polarity", x= "Longitud", y="Sentimiento", fill="Grupo") + theme_minimal() 
review_box <- ggplot(data = review, aes(x=sentimiento, y= longitud, fill=sentimiento)) + geom_boxplot(fill= c("#A23B72","#2E86AB")) + labs(title = "Review_polarity", x= "Sentimiento", y="Longitud") + theme_minimal() + theme(legend.position = "none")

#conjunto
conjun_densi <-ggplot(data =dua, aes(x=longitud, fill=sentimiento)) + geom_density(alpha = 0.35) + scale_fill_manual(values = c("pos"="#2E86AB", "neg"="#A23B72"))+ scale_color_manual(values = c("pos"="#2E86AB", "neg"="#A23B72")) + labs(title = "Unificado", x= "Longitud", y="Sentimiento", fill="Grupo") + theme_minimal()
conjun_box <-ggplot(data = dua, aes(x=sentimiento, y= longitud, fill=sentimiento)) + geom_boxplot(fill= c("#A23B72","#2E86AB")) + labs(title = "Unificado", x= "Sentimiento", y="Longitud") + theme_minimal() + theme(legend.position = "none")

densidad_conjunta <- (amd_densi+amd_box)/(stan_densi+stan_box)/(review_densi+review_box)/(conjun_densi+conjun_box) + plot_annotation(title = "Comparación de distribuciones", theme = theme(plot.title = element_text(hjust = 0.5, size = 16))) & theme(plot.margin = margin(10, 10, 10, 10)) 


ggsave("4-grafico_densidad.png", plot=densidad_conjunta, height = 10, width = 10)

# test ManWhitten U 
wilcox.test(longitud ~ sentimiento, data = amd, alternative = "two.sided", conf.int = TRUE, exact = FALSE)  
wilcox.test(longitud ~ sentimiento, data = stan, alternative = "two.sided", conf.int = TRUE, exact = FALSE)  
wilcox.test(longitud ~ sentimiento, data = review, alternative = "two.sided", conf.int = TRUE, exact = FALSE)  
wilcox.test(longitud ~ sentimiento, data = dua, alternative = "two.sided", conf.int = TRUE, exact = FALSE)  



#Violin plots y tabla de estadisticos

tabla_estadisticos_vocabulario <- dua %>% group_by(dataset) %>% summarise(
    n = n(),
    media = mean(vocabulario, na.rm = TRUE),
    mediana = median(vocabulario, na.rm = TRUE),
    sd = sd(vocabulario, na.rm = TRUE),
    min = min(vocabulario, na.rm = TRUE),
    max = max(vocabulario, na.rm = TRUE),
    q1 = quantile(vocabulario, 0.25, na.rm = TRUE),
    q3 = quantile(vocabulario, 0.75, na.rm = TRUE),
    iqr = IQR(vocabulario, na.rm = TRUE)
  )

tabla_estadisticos_caracteres <- dua %>% group_by(dataset) %>% summarise(
  n = n(),
  media = mean(num_caracteres, na.rm = TRUE),
  mediana = median(num_caracteres, na.rm = TRUE),
  sd = sd(num_caracteres, na.rm = TRUE),
  min = min(num_caracteres, na.rm = TRUE),
  max = max(num_caracteres, na.rm = TRUE),
  q1 = quantile(num_caracteres, 0.25, na.rm = TRUE),
  q3 = quantile(num_caracteres, 0.75, na.rm = TRUE),
  iqr = IQR(num_caracteres, na.rm = TRUE)
)

tabla_estadisticos_longitud <- dua %>% group_by(dataset) %>% summarise(
  n = n(),
  media = mean(longitud, na.rm = TRUE),
  mediana = median(longitud, na.rm = TRUE),
  sd = sd(longitud, na.rm = TRUE),
  min = min(longitud, na.rm = TRUE),
  max = max(longitud, na.rm = TRUE),
  q1 = quantile(longitud, 0.25, na.rm = TRUE),
  q3 = quantile(longitud, 0.75, na.rm = TRUE),
  iqr = IQR(longitud, na.rm = TRUE)
)


tabla_estadisticos_vocabulario
tabla_estadisticos_longitud
tabla_estadisticos_caracteres

#Longitud de palabras
long <- ggplot(data=dua, aes(x=dataset, y=longitud, fill = dataset ) ) + geom_violin(width = 0.7, scale = "width", trim = TRUE, alpha = 0.8) + labs(title = "Longitud de reseñas por dataset") + theme_minimal() + theme(legend.position = "none")

# Diversidad lexica
vocab <- ggplot(data=dua, aes(x=dataset, y=vocabulario, fill = dataset )) + geom_violin(width = 0.7, scale = "width", trim = TRUE, alpha = 0.8) + labs(title = "Vocabulario usado por dataset") + theme_minimal() + theme(legend.position = "none")

#Numero de palabras
num_c <- ggplot(data=dua, aes(x=dataset, y=num_caracteres, fill = dataset )) + geom_violin(width = 0.7, scale = "width", trim = TRUE, alpha = 0.8) + labs(title = "Numero de palabras por dataset")  + theme_minimal() + theme(legend.position = "none")


# Matriz de correlacion
library(tidyverse)
library(corrplot)
numeric <- dua %>% select(where(is.numeric))

matriz_cor <- cor(numeric)

png("7-matriz_correlacion.png", width = 10, height = 10, units = "in", res = 300)

corrplot(matriz_cor, 
         method = "square",      # circle, square, ellipse, number, color
         type = "lower",         # full, upper, lower
         tl.col = "black",       # color de labels
         tl.srt = 45,            # rotación de labels
         addCoef.col = "white",  # agregar coeficientes
         number.cex = 0.7,       # tamaño de números
         tl.cex = 0.9,)           # tamaño de labels

dev.off()
