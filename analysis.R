# =============================================================================
# АНАЛИЗ ДАННЫХ: УСТОЙЧИВОЕ ПОТРЕБЛЕНИЕ (ЭКОЛОГИЧНОЕ ПОВЕДЕНИЕ)
# =============================================================================
# Версия без tidyverse — только base R + car (минимальные зависимости)
# Запуск: Rscript analysis.R  или  source("analysis.R") в R
# Результаты: results_summary.txt, папка plots/
# =============================================================================

# Пакет car (для VIF) — опционально. Без интернета/CRAN работает без него.
has_car <- requireNamespace("car", quietly = TRUE)

# Создание папки для графиков
dir.create("plots", showWarnings = FALSE)

# Настройка вывода
sink("results_summary.txt", split = TRUE)

# =============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =============================================================================
cat("\n========== 1. ЗАГРУЗКА ДАННЫХ ==========\n\n")

load_data <- function() {
  if (file.exists("document.csv")) {
    # Кодировка: cp1251 типична для русского Windows, UTF-8 — для экспортов
    encodings <- c("cp1251", "UTF-8", "latin1")
    df_raw <- NULL
    for (enc in encodings) {
      df_raw <- tryCatch({
        read.csv2("document.csv", sep = ";", dec = ",", quote = "\"",
                  skip = 2, header = FALSE, fill = TRUE,
                  fileEncoding = enc, stringsAsFactors = FALSE,
                  na.strings = c("", "NA"), strip.white = TRUE)
      }, error = function(e) NULL)
      if (!is.null(df_raw) && nrow(df_raw) > 0) {
        cat("Загружен document.csv (кодировка:", enc, ")\n")
        return(df_raw)
      }
    }
    # Без указания кодировки (системная по умолчанию)
    df_raw <- tryCatch({
      read.csv2("document.csv", sep = ";", dec = ",", quote = "\"",
                skip = 2, header = FALSE, fill = TRUE,
                stringsAsFactors = FALSE, na.strings = c("", "NA"))
    }, error = function(e) NULL)
    if (is.null(df_raw) || nrow(df_raw) == 0)
      stop("Не удалось прочитать document.csv. Проверьте путь и кодировку файла.")
    cat("Загружен document.csv\n")
    return(df_raw)
  }
  if (file.exists("document.xlsx")) {
    if (!requireNamespace("readxl", quietly = TRUE))
      stop("Нужен пакет readxl. Установите: install.packages('readxl')")
    df_raw <- as.data.frame(readxl::read_excel("document.xlsx"))
    cat("Загружен document.xlsx\n")
    return(df_raw)
  }
  stop("Не найден document.csv или document.xlsx")
}

df_raw <- load_data()

col_names <- c("Timestamp", "age", "gender", "education", "living", "city", "income",
               "q2_1_waste", "q2_2_eco", "q2_3_energy",
               "q3_1_lamp", "q3_2_contribution", "q3_3_store", "q3_4_consumption", "q3_5_logic",
               "q4_1_friends", "q4_2_neighbors", "q4_3_opinion",
               "q5_1_barriers", "barriers_text", "contacts")

if (ncol(df_raw) >= length(col_names)) {
  names(df_raw)[1:length(col_names)] <- col_names
  df <- df_raw[, 1:length(col_names)]
} else {
  df <- df_raw
  names(df)[1:ncol(df)] <- col_names[1:ncol(df)]
}

# Функция для генерации
score_freq <- function(x) {
  ifelse(grepl("Никогда", x), 1,
  ifelse(grepl("Редко", x), 2,
  ifelse(grepl("Иногда", x), 3,
  ifelse(grepl("Часто", x), 4,
  ifelse(grepl("Постоянно", x), 5, NA)))))
}

df$generation <- ifelse(grepl("18-22", df$age), "zoom",
                 ifelse(grepl("23-28|29-35|35\\+", df$age), "millennial", "other"))

cat("Размер выборки: n =", nrow(df), "\n")

# =============================================================================
# 2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА И КРОСС-ТАБУЛЯЦИЯ
# =============================================================================
cat("\n\n========== 2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ==========\n\n")

cat("--- Частоты: Сортировка отходов (2.1) ---\n")
print(table(df$q2_1_waste, useNA = "ifany"))

cat("\n--- Частоты: Поиск эко-товаров (2.2) ---\n")
print(table(df$q2_2_eco, useNA = "ifany"))

cat("\n--- Частоты: Экономия воды/электричества (2.3) ---\n")
print(table(df$q2_3_energy, useNA = "ifany"))

cat("\n--- Распределение по возрасту ---\n")
print(table(df$age))

cat("\n--- Кросс-табуляция: Сортировка x Возраст ---\n")
t1 <- table(df$age, df$q2_1_waste)
print(addmargins(t1))

cat("\n--- Кросс-табуляция: Сортировка x Доход ---\n")
t2 <- table(df$income, df$q2_1_waste)
print(t2)

cat("\n--- Кросс-табуляция: Сортировка x Образование ---\n")
t3 <- table(df$education, df$q2_1_waste)
print(t3)

cat("\n--- Кросс-табуляция: Эко-товары x Возраст ---\n")
t4 <- table(df$age, df$q2_2_eco)
print(addmargins(t4))

cat("\n--- Кросс-табуляция: Эко-товары x Доход ---\n")
t5 <- table(df$income, df$q2_2_eco)
print(t5)

cat("\n--- Кросс-табуляция: Эко-товары x Образование ---\n")
t6 <- table(df$education, df$q2_2_eco)
print(t6)

cat("\n--- Кросс-табуляция: Экономия x Возраст ---\n")
t7 <- table(df$age, df$q2_3_energy)
print(addmargins(t7))

cat("\n--- Кросс-табуляция: Экономия x Доход ---\n")
t8 <- table(df$income, df$q2_3_energy)
print(t8)

cat("\n--- Кросс-табуляция: Экономия x Образование ---\n")
t9 <- table(df$education, df$q2_3_energy)
print(t9)

# Визуализация 1: Сводный barplot возраст x частота сортировки
t1_rel <- prop.table(t1, 1) * 100
png("plots/01_crosstab_age_waste.png", width = 10, height = 6, units = "in", res = 300)
par(mar = c(8, 4, 4, 2))
barplot(t(t1_rel), beside = FALSE, las = 2, col = gray.colors(ncol(t1_rel)),
        main = "Распределение практики сортировки мусора по возрасту (%)",
        xlab = "", ylab = "Доля (%)", legend.text = TRUE, args.legend = list(x = "topright"))
dev.off()

# Визуализация 2: Доход x сортировка
t2_rel <- prop.table(t2, 1) * 100
png("plots/02_crosstab_income_waste.png", width = 10, height = 6, units = "in", res = 300)
par(mar = c(10, 4, 4, 2))
barplot(t(t2_rel), beside = FALSE, las = 2, col = gray.colors(ncol(t2_rel)),
        main = "Распределение практики сортировки мусора по доходу (%)",
        xlab = "", ylab = "Доля (%)", legend.text = TRUE, args.legend = list(x = "topright"))
dev.off()

# =============================================================================
# 3. СТАТИСТИЧЕСКИЕ ТЕСТЫ
# =============================================================================
cat("\n\n========== 3. СТАТИСТИЧЕСКИЕ ТЕСТЫ ==========\n\n")

# --- 3.1. Хи-квадрат тесты (все комбинации) ---
cat("=== ХИ-КВАДРАТ ТЕСТЫ (все комбинации) ===\n\n")

cat("--- Хи-квадрат: Возраст x Сортировка ---\n")
print(chisq.test(t1))

cat("\n--- Хи-квадрат: Возраст x Эко-товары ---\n")
print(chisq.test(t4))

cat("\n--- Хи-квадрат: Возраст x Экономия ---\n")
print(chisq.test(t7))

cat("\n--- Хи-квадрат: Доход x Сортировка ---\n")
print(chisq.test(t2))

cat("\n--- Хи-квадрат: Доход x Эко-товары ---\n")
print(chisq.test(t5))

cat("\n--- Хи-квадрат: Доход x Экономия ---\n")
print(chisq.test(t8))

cat("\n--- Хи-квадрат: Образование x Сортировка ---\n")
print(chisq.test(t3))

cat("\n--- Хи-квадрат: Образование x Эко-товары ---\n")
print(chisq.test(t6))

cat("\n--- Хи-квадрат: Образование x Экономия ---\n")
print(chisq.test(t9))

# --- 3.2. U-критерий Манна–Уитни / Краскела–Уоллиса (все комбинации) ---
cat("\n\n=== U-КРИТЕРИЙ МАННА–УИТНИ / КРАСКЕЛА–УОЛЛИСА (все комбинации) ===\n\n")

df$q2_1_num <- score_freq(df$q2_1_waste)
df$q2_2_num <- score_freq(df$q2_2_eco)
df$q2_3_num <- score_freq(df$q2_3_energy)

# --- Возраст: Зумеры vs Миллениалы (2 группы → Манн–Уитни) ---
df_mw <- df[df$generation %in% c("zoom", "millennial"), ]

cat("--- Манн–Уитни: Возраст (Зумеры vs Миллениалы) x Сортировка ---\n")
w <- wilcox.test(q2_1_num ~ generation, data = df_mw, exact = FALSE)
cat("W =", w$statistic, ", p =", w$p.value, "\n")

cat("\n--- Манн–Уитни: Возраст (Зумеры vs Миллениалы) x Эко-товары ---\n")
w <- wilcox.test(q2_2_num ~ generation, data = df_mw, exact = FALSE)
cat("W =", w$statistic, ", p =", w$p.value, "\n")

cat("\n--- Манн–Уитни: Возраст (Зумеры vs Миллениалы) x Экономия ---\n")
w <- wilcox.test(q2_3_num ~ generation, data = df_mw, exact = FALSE)
cat("W =", w$statistic, ", p =", w$p.value, "\n")

# --- Доход (>2 групп → Краскел–Уоллис + попарные Манн–Уитни) ---
cat("\n--- Краскел–Уоллис: Доход x Сортировка ---\n")
kw <- kruskal.test(q2_1_num ~ income, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
print(pairwise.wilcox.test(df$q2_1_num, df$income, p.adjust.method = "bonferroni", exact = FALSE))

cat("\n--- Краскел–Уоллис: Доход x Эко-товары ---\n")
kw <- kruskal.test(q2_2_num ~ income, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
print(pairwise.wilcox.test(df$q2_2_num, df$income, p.adjust.method = "bonferroni", exact = FALSE))

cat("\n--- Краскел–Уоллис: Доход x Экономия ---\n")
kw <- kruskal.test(q2_3_num ~ income, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
print(pairwise.wilcox.test(df$q2_3_num, df$income, p.adjust.method = "bonferroni", exact = FALSE))

# --- Образование (>2 групп → Краскел–Уоллис + попарные Манн–Уитни) ---
cat("\n--- Краскел–Уоллис: Образование x Сортировка ---\n")
kw <- kruskal.test(q2_1_num ~ education, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
print(pairwise.wilcox.test(df$q2_1_num, df$education, p.adjust.method = "bonferroni", exact = FALSE))

cat("\n--- Краскел–Уоллис: Образование x Эко-товары ---\n")
kw <- kruskal.test(q2_2_num ~ education, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
print(pairwise.wilcox.test(df$q2_2_num, df$education, p.adjust.method = "bonferroni", exact = FALSE))

cat("\n--- Краскел–Уоллис: Образование x Экономия ---\n")
kw <- kruskal.test(q2_3_num ~ education, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
print(pairwise.wilcox.test(df$q2_3_num, df$education, p.adjust.method = "bonferroni", exact = FALSE))

# =============================================================================
# 4. ИНДЕКС УСТОЙЧИВОГО ПОВЕДЕНИЯ
# =============================================================================
cat("\n\n========== 4. ИНДЕКС УСТОЙЧИВОГО ПОВЕДЕНИЯ ==========\n\n")

df$q2_1_score <- score_freq(df$q2_1_waste)
df$q2_2_score <- score_freq(df$q2_2_eco)
df$q2_3_score <- score_freq(df$q2_3_energy)
df$sustainable_index <- df$q2_1_score + df$q2_2_score + df$q2_3_score

cat("Описательная статистика индекса (диапазон 3–15):\n")
print(summary(df$sustainable_index))
cat("SD =", round(sd(df$sustainable_index, na.rm = TRUE), 2), ", N =", sum(!is.na(df$sustainable_index)), "\n")

# Визуализация 3: Гистограмма
png("plots/03_histogram_index.png", width = 8, height = 5, units = "in", res = 300)
hist(df$sustainable_index[!is.na(df$sustainable_index)], breaks = seq(2.5, 15.5, 1),
     col = "steelblue", main = "Распределение индекса устойчивого поведения",
     xlab = "Индекс (3–15)", ylab = "Частота")
dev.off()

# Визуализация 4: Boxplot по поколениям
df_bp <- df[df$generation %in% c("zoom", "millennial") & !is.na(df$sustainable_index), ]
png("plots/04_boxplot_generation.png", width = 6, height = 5, units = "in", res = 300)
boxplot(sustainable_index ~ generation, data = df_bp, col = c("lightgreen", "lightblue"),
        names = c("Миллениалы (23-35+)", "Зумеры (18-22)"),
        main = "Индекс устойчивого поведения по поколениям", ylab = "Индекс")
dev.off()

# =============================================================================
# 5. ПОДГОТОВКА ПРЕДИКТОРОВ
# =============================================================================
cat("\n\n========== 5. ПРЕДИКТОРЫ ==========\n\n")

df$q3_1_led <- as.numeric(grepl("Опция Б|LED", df$q3_1_lamp))
df$q3_2_agree <- as.numeric(grepl("Согласен", df$q3_2_contribution))
df$q3_3_willing <- as.numeric(grepl("Да, если", df$q3_3_store))

df$q4_1_num <- ifelse(grepl("Почти никто", df$q4_1_friends), 1,
               ifelse(grepl("Немного", df$q4_1_friends), 2,
               ifelse(grepl("Половина", df$q4_1_friends), 3,
               ifelse(grepl("Большинство", df$q4_1_friends), 4,
               ifelse(grepl("Почти все", df$q4_1_friends), 5, NA)))))

df$q4_2_num <- ifelse(grepl("Очень маловероятно", df$q4_2_neighbors), 1,
               ifelse(grepl("Скорее нет", df$q4_2_neighbors), 2,
               ifelse(grepl("Нейтрально|затрудняюсь", df$q4_2_neighbors), 3,
               ifelse(grepl("Скорее да", df$q4_2_neighbors), 4,
               ifelse(grepl("Очень вероятно|все делают", df$q4_2_neighbors), 5, NA)))))

df$q4_3_num <- ifelse(grepl("Не важно вообще|только для себя", df$q4_3_opinion), 1,
               ifelse(grepl("Слабо важно", df$q4_3_opinion), 2,
               ifelse(grepl("Нейтрально", df$q4_3_opinion), 3,
               ifelse(grepl("Довольно важно", df$q4_3_opinion), 4,
               ifelse(grepl("Очень важно|тренде", df$q4_3_opinion), 5, NA)))))

df$barrier_infra <- as.numeric(grepl("Нет инфраструктуры|инфраструктур", df$q5_1_barriers, ignore.case = TRUE))
df$barrier_lazy <- as.numeric(grepl("Лень|нет времени|забываю", df$q5_1_barriers, ignore.case = TRUE))
df$barrier_unbelief <- as.numeric(grepl("Не верю", df$q5_1_barriers))
df$barrier_inconvenient <- as.numeric(grepl("неудобно|специальные магазины", df$q5_1_barriers, ignore.case = TRUE))
df$barrier_expense <- as.numeric(grepl("дорого|нет денег", df$q5_1_barriers, ignore.case = TRUE))

df$income_f <- factor(df$income)
df$education_f <- factor(df$education)

# =============================================================================
# 6. МНОЖЕСТВЕННАЯ РЕГРЕССИЯ
# =============================================================================
cat("\n\n========== 6. МНОЖЕСТВЕННАЯ РЕГРЕССИЯ ==========\n\n")

model <- lm(sustainable_index ~ q3_1_led + q3_2_agree + q3_3_willing +
              q4_1_num + q4_2_num + q4_3_num + income_f + education_f +
              barrier_infra + barrier_lazy + barrier_unbelief + barrier_inconvenient + barrier_expense,
            data = df, na.action = na.exclude)

print(summary(model))

if (has_car) {
  cat("\n--- VIF (мультиколлинеарность) ---\n")
  tryCatch(print(car::vif(model)), error = function(e) cat("VIF не вычислен\n"))
}

# Диагностика остатков
png("plots/05_residuals_diagnostics.png", width = 12, height = 5, units = "in", res = 300)
par(mfrow = c(1, 2))
plot(model, which = 1)
plot(model, which = 2)
dev.off()

# =============================================================================
# 7. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ПОКОЛЕНИЙ
# =============================================================================
cat("\n\n========== 7. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ПОКОЛЕНИЙ ==========\n\n")

df_gen <- df[df$generation %in% c("zoom", "millennial"), ]
df_zoom <- df_gen[df_gen$generation == "zoom", ]
df_mill <- df_gen[df_gen$generation == "millennial", ]

cat("--- Регрессия: ЗУМЕРЫ ---\n")
model_zoom <- lm(sustainable_index ~ q4_1_num + q4_2_num + q4_3_num +
                   barrier_infra + barrier_lazy + barrier_unbelief + barrier_inconvenient + barrier_expense + income_f,
                 data = df_zoom, na.action = na.exclude)
print(summary(model_zoom))

cat("\n--- Регрессия: МИЛЛЕНИАЛЫ ---\n")
model_mill <- lm(sustainable_index ~ q4_1_num + q4_2_num + q4_3_num +
                   barrier_infra + barrier_lazy + barrier_unbelief + barrier_inconvenient + barrier_expense + income_f,
                 data = df_mill, na.action = na.exclude)
print(summary(model_mill))

cat("\n--- Модель с взаимодействиями ---\n")
df_gen$generation <- factor(df_gen$generation)
model_int <- lm(sustainable_index ~ generation * (barrier_infra + barrier_lazy + barrier_unbelief + 
                barrier_inconvenient + barrier_expense) + q4_1_num + q4_2_num + q4_3_num + income_f,
                data = df_gen, na.action = na.exclude)
print(summary(model_int))

# Визуализация 6: Сравнение коэффициентов барьеров
coef_z <- coef(model_zoom)
coef_m <- coef(model_mill)
bn <- c("barrier_infra", "barrier_lazy", "barrier_unbelief", "barrier_inconvenient", "barrier_expense")
labels <- c("Нет инфраструктуры", "Лень/время", "Не верю в пользу", "Неудобно", "Дорого")
vals <- numeric(length(bn) * 2)
for (i in seq_along(bn)) {
  vals[i] <- if (bn[i] %in% names(coef_z)) coef_z[bn[i]] else NA
  vals[length(bn) + i] <- if (bn[i] %in% names(coef_m)) coef_m[bn[i]] else NA
}
coef_mat <- matrix(vals, nrow = length(bn), ncol = 2, dimnames = list(labels, c("zoom", "millennial")))

png("plots/06_coefficients_generation.png", width = 8, height = 5, units = "in", res = 300)
par(mar = c(4, 10, 4, 2))
barplot(t(coef_mat), beside = TRUE, horiz = TRUE, col = c("lightgreen", "lightblue"),
        legend.text = c("Зумеры", "Миллениалы"), args.legend = list(x = "bottomright"),
        main = "Сравнение коэффициентов барьеров по поколениям",
        xlab = "Коэффициент")
dev.off()

# =============================================================================
# 8. ИТОГИ
# =============================================================================
cat("\n\n========== ИТОГИ ==========\n\n")
cat("Выборка: n =", nrow(df), "\n")
cat("Индекс: M =", round(mean(df$sustainable_index, na.rm = TRUE), 2),
    ", SD =", round(sd(df$sustainable_index, na.rm = TRUE), 2), "\n")
cat("R² модели:", round(summary(model)$r.squared, 3), "\n")
cat("Графики сохранены в plots/\n")

sink()
cat("\nГотово. Результаты в results_summary.txt, графики в plots/\n")
