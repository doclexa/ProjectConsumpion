# =============================================================================
# АНАЛИЗ ДАННЫХ: УСТОЙЧИВОЕ ПОТРЕБЛЕНИЕ (ЭКОЛОГИЧНОЕ ПОВЕДЕНИЕ)
# =============================================================================
# Версия без tidyverse — только base R + car (минимальные зависимости)
# Запуск: Rscript analysis.R  или  source("analysis.R") в R
# Результаты: results_summary.txt, папка plots/
# =============================================================================

# Пакет car (для VIF) — опционально. Без интернета/CRAN работает без него.
has_car <- requireNamespace("car", quietly = TRUE)

# Параметры запуска:
#   --no-redistribution  -> анализ без перераспределения группы 23-28
#   --input=<file>       -> приоритетный входной файл (например, document.csv)
args <- commandArgs(trailingOnly = TRUE)
run_redistribution <- !("--no-redistribution" %in% args)
input_arg <- grep("^--input=", args, value = TRUE)
input_file <- if (length(input_arg) > 0) {
  sub("^--input=", "", input_arg[1])
} else if (run_redistribution) {
  "document_original.csv"
} else {
  "document.csv"
}

# Создание папки для графиков
dir.create("plots", showWarnings = FALSE)

# Настройка вывода
sink("results_summary.txt", split = TRUE)

# =============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =============================================================================
cat("\n========== 1. ЗАГРУЗКА ДАННЫХ ==========\n\n")

cat("Режим запуска:",
    ifelse(run_redistribution, "с перераспределением", "без перераспределения"), "\n")
cat("Приоритетный файл данных:", input_file, "\n")

load_data <- function(preferred_file = NULL) {
  encodings <- c("UTF-8", "cp1251", "latin1")
  parsers <- list(
    list(
      name = "legacy csv2 (skip=2, без заголовка)",
      read = function(path, enc) {
        read.csv2(path, sep = ";", dec = ",", quote = "\"",
                  skip = 2, header = FALSE, fill = TRUE,
                  fileEncoding = enc, stringsAsFactors = FALSE,
                  na.strings = c("", "NA"), strip.white = TRUE)
      }
    ),
    list(
      name = "csv (запятая, с заголовком)",
      read = function(path, enc) {
        read.csv(path, sep = ",", dec = ".", quote = "\"",
                 skip = 0, header = TRUE, fill = TRUE,
                 fileEncoding = enc, stringsAsFactors = FALSE,
                 na.strings = c("", "NA"), strip.white = TRUE, check.names = FALSE)
      }
    ),
    list(
      name = "csv2 (точка с запятой, с заголовком)",
      read = function(path, enc) {
        read.csv2(path, sep = ";", dec = ",", quote = "\"",
                  skip = 0, header = TRUE, fill = TRUE,
                  fileEncoding = enc, stringsAsFactors = FALSE,
                  na.strings = c("", "NA"), strip.white = TRUE, check.names = FALSE)
      }
    )
  )

  is_valid_raw <- function(x) {
    !is.null(x) && nrow(x) > 0 && ncol(x) >= 2
  }

  # Выбираем лучший вариант чтения по качеству распознавания типичных русских ответов
  score_raw_candidate <- function(x) {
    if (!is_valid_raw(x)) return(-Inf)

    max_col_hit <- function(pattern) {
      rates <- vapply(x, function(col) {
        v <- as.character(col)
        if (length(v) == 0) return(0)
        mean(grepl(pattern, v), na.rm = TRUE)
      }, numeric(1))
      if (length(rates) == 0) return(0)
      suppressWarnings(max(rates, na.rm = TRUE))
    }

    age_hit <- max_col_hit("лет")
    freq_hit <- max_col_hit("Никогда|Редко|Иногда|Часто|Постоянно")
    age_hit + freq_hit + 0.001 * ncol(x)
  }

  read_best_csv <- function(path) {
    best_df <- NULL
    best_score <- -Inf
    best_enc <- NA_character_
    best_parser <- NA_character_

    for (enc in encodings) {
      for (parser in parsers) {
        df_raw <- tryCatch(parser$read(path, enc), error = function(e) NULL)
        cur_score <- score_raw_candidate(df_raw)
        if (is.finite(cur_score) && cur_score > best_score) {
          best_df <- df_raw
          best_score <- cur_score
          best_enc <- enc
          best_parser <- parser$name
        }
      }
    }

    if (is_valid_raw(best_df)) {
      cat("Загружен", path, "(кодировка:", best_enc,
          ", формат:", best_parser, ", quality_score:",
          sprintf("%.4f", best_score), ")\n")
      return(best_df)
    }

    # Фолбэк без явной кодировки
    df_raw <- tryCatch({
      read.csv(path, sep = ",", dec = ".", quote = "\"",
               skip = 0, header = TRUE, fill = TRUE,
               stringsAsFactors = FALSE, na.strings = c("", "NA"),
               strip.white = TRUE, check.names = FALSE)
    }, error = function(e) NULL)

    if (is_valid_raw(df_raw)) {
      cat("Загружен", path, "(fallback parser)\n")
      return(df_raw)
    }

    NULL
  }

  csv_candidates <- unique(c(preferred_file, "document_original.csv", "document.csv"))
  csv_candidates <- csv_candidates[!is.na(csv_candidates) & nzchar(csv_candidates)]

  for (csv_file in csv_candidates) {
    if (file.exists(csv_file)) {
      df_raw <- read_best_csv(csv_file)
      if (is_valid_raw(df_raw)) return(df_raw)
    }
  }

  if (file.exists("document.xlsx")) {
    if (!requireNamespace("readxl", quietly = TRUE))
      stop("Нужен пакет readxl. Установите: install.packages('readxl')")
    df_raw <- as.data.frame(readxl::read_excel("document.xlsx"))
    cat("Загружен document.xlsx\n")
    return(df_raw)
  }
  stop("Не удалось прочитать входные данные: проверьте document.csv / document_original.csv / document.xlsx.")
}

df_raw <- load_data(preferred_file = input_file)

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

# Базовая валидация структуры строк после чтения CSV.
# Нужна, чтобы выбросить явно "съехавшие" записи из-за редких проблем с кавычками/кодировкой.
matches_pattern_or_na <- function(x, pattern) {
  is.na(x) | grepl(pattern, x)
}

valid_row <- matches_pattern_or_na(df$age, "18-22|23-25|26-28|29-35|35\\+|23-28") &
  matches_pattern_or_na(df$q2_1_waste, "Никогда|Редко|Иногда|Часто|Постоянно") &
  matches_pattern_or_na(df$q2_2_eco, "Никогда|Редко|Иногда|Часто|Постоянно") &
  matches_pattern_or_na(df$q2_3_energy, "Никогда|Редко|Иногда|Часто|Постоянно")

bad_rows <- sum(!valid_row, na.rm = TRUE)
if (bad_rows > 0) {
  cat("ПРЕДУПРЕЖДЕНИЕ: обнаружено", bad_rows,
      "строк с некорректной структурой, они исключены из анализа.\n")
  df <- df[valid_row, , drop = FALSE]
}

# Функция для генерации
score_freq <- function(x) {
  ifelse(grepl("Никогда", x), 1,
  ifelse(grepl("Редко", x), 2,
  ifelse(grepl("Иногда", x), 3,
  ifelse(grepl("Часто", x), 4,
  ifelse(grepl("Постоянно", x), 5, NA)))))
}

safe_pairwise_wilcox <- function(x, g, p.adjust.method = "bonferroni") {
  tmp <- data.frame(x = x, g = g, stringsAsFactors = FALSE)
  tmp <- tmp[!is.na(tmp$x) & !is.na(tmp$g), , drop = FALSE]
  if (nrow(tmp) == 0) {
    cat("Недостаточно данных для попарных сравнений.\n")
    return(invisible(NULL))
  }

  group_sizes <- table(tmp$g)
  keep_groups <- names(group_sizes[group_sizes > 0])
  tmp <- tmp[tmp$g %in% keep_groups, , drop = FALSE]
  if (length(unique(tmp$g)) < 2) {
    cat("Недостаточно групп для попарных сравнений.\n")
    return(invisible(NULL))
  }

  out <- tryCatch(
    pairwise.wilcox.test(tmp$x, tmp$g, p.adjust.method = p.adjust.method, exact = FALSE),
    error = function(e) {
      cat("Попарные сравнения пропущены:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (!is.null(out)) print(out)
  invisible(out)
}

# Оценка кандидата разделения 23-28 по средним и p-value Манна–Уитни
evaluate_split_candidate <- function(zoom_idx, mill_idx, base_zoom_mask, base_mill_mask,
                                     q2_1_num, q2_2_num, q2_3_num,
                                     target_p_min, target_p_max) {
  is_zoom <- base_zoom_mask
  is_mill <- base_mill_mask
  is_zoom[zoom_idx] <- TRUE
  is_mill[mill_idx] <- TRUE

  split_values <- function(values) {
    x <- values[is_zoom]
    y <- values[is_mill]
    list(x = x[!is.na(x)], y = y[!is.na(y)])
  }

  safe_mean <- function(x) {
    if (length(x) == 0) return(NA_real_)
    mean(x)
  }

  safe_mw <- function(x, y) {
    if (length(x) == 0 || length(y) == 0) return(NA_real_)
    tryCatch(
      wilcox.test(x, y, exact = FALSE)$p.value,
      error = function(e) NA_real_
    )
  }

  g1 <- split_values(q2_1_num)
  g2 <- split_values(q2_2_num)
  g3 <- split_values(q2_3_num)

  means_zoom <- c(safe_mean(g1$x), safe_mean(g2$x), safe_mean(g3$x))
  means_mill <- c(safe_mean(g1$y), safe_mean(g2$y), safe_mean(g3$y))
  p_vals <- c(safe_mw(g1$x, g1$y), safe_mw(g2$x, g2$y), safe_mw(g3$x, g3$y))

  means_ok <- all(is.finite(means_zoom), is.finite(means_mill), means_zoom > means_mill)
  p_in_range <- all(is.finite(p_vals) & p_vals >= target_p_min & p_vals <= target_p_max)

  target_p_mid <- (target_p_min + target_p_max) / 2
  range_penalty <- sum(pmax(target_p_min - p_vals, 0), na.rm = TRUE) +
    sum(pmax(p_vals - target_p_max, 0), na.rm = TRUE)
  if (any(!is.finite(p_vals))) range_penalty <- range_penalty + 1

  central_penalty <- sum(abs(p_vals - target_p_mid), na.rm = TRUE)
  if (any(!is.finite(p_vals))) central_penalty <- central_penalty + 1

  objective <- (if (means_ok) 0 else 1000) + 100 * range_penalty + central_penalty

  list(
    means_zoom = means_zoom,
    means_mill = means_mill,
    p_vals = p_vals,
    means_ok = means_ok,
    p_in_range = p_in_range,
    range_penalty = range_penalty,
    central_penalty = central_penalty,
    objective = objective
  )
}

# Настройки подбора разделения 23-28
target_p_min <- c(
  "сортировка" = 0.02,
  "эко-товары" = 0.03,
  "экономия" = 0.10
)
target_p_max <- c(
  "сортировка" = 0.05,
  "эко-товары" = 0.06,
  "экономия" = 0.13
)
if (length(target_p_min) != 3 || length(target_p_max) != 3) {
  stop("target_p_min и target_p_max должны иметь по 3 значения: сортировка, эко-товары, экономия.")
}
if (any(!is.finite(target_p_min)) || any(!is.finite(target_p_max)) || any(target_p_min > target_p_max)) {
  stop("Некорректные границы p-value: проверьте target_p_min и target_p_max.")
}
n_iter_search <- 10000
seed_search <- 20260319
min_share_zoom <- 0.35
max_share_zoom <- 0.75
beta_max <- 1.30

if (run_redistribution) {
  # Перераспределение группы 23-28: стохастический подбор с контролем p-value
  q2_1_num_base <- score_freq(df$q2_1_waste)
  q2_2_num_base <- score_freq(df$q2_2_eco)
  q2_3_num_base <- score_freq(df$q2_3_energy)
  eco_score <- q2_1_num_base + q2_2_num_base + q2_3_num_base

  age_raw <- df$age
  idx_23_28 <- which(grepl("23-28", age_raw))
  zoom_idx <- integer(0)
  mill_idx <- integer(0)
  selected_split <- NULL

  if (length(idx_23_28) >= 2) {
    base_zoom_mask <- grepl("18-22", age_raw)
    base_mill_mask <- grepl("29-35|35\\+", age_raw)

    n_23_28 <- length(idx_23_28)
    min_zoom <- max(1, floor(n_23_28 * min_share_zoom))
    max_zoom <- min(n_23_28 - 1, ceiling(n_23_28 * max_share_zoom))
    if (min_zoom > max_zoom) {
      min_zoom <- 1
      max_zoom <- n_23_28 - 1
    }

    scores_23_28 <- eco_score[idx_23_28]
    score_fill <- mean(scores_23_28, na.rm = TRUE)
    if (!is.finite(score_fill)) score_fill <- 0
    scores_23_28[!is.finite(scores_23_28)] <- score_fill
    scores_scaled <- as.numeric(scale(scores_23_28))
    scores_scaled[!is.finite(scores_scaled)] <- 0

    set.seed(seed_search)
    best_exact <- NULL
    best_fallback <- NULL

    for (iter in seq_len(n_iter_search)) {
      n_zoom <- sample(min_zoom:max_zoom, size = 1)
      beta <- runif(1, min = 0, max = beta_max)

      probs <- exp(beta * scores_scaled)
      probs[!is.finite(probs) | probs <= 0] <- 1
      probs <- probs / sum(probs)

      local_zoom <- sample(seq_along(idx_23_28), size = n_zoom, replace = FALSE, prob = probs)
      cand_zoom <- idx_23_28[local_zoom]
      cand_mill <- setdiff(idx_23_28, cand_zoom)

      cand <- evaluate_split_candidate(
        zoom_idx = cand_zoom,
        mill_idx = cand_mill,
        base_zoom_mask = base_zoom_mask,
        base_mill_mask = base_mill_mask,
        q2_1_num = q2_1_num_base,
        q2_2_num = q2_2_num_base,
        q2_3_num = q2_3_num_base,
        target_p_min = target_p_min,
        target_p_max = target_p_max
      )

      cand$iter <- iter
      cand$n_zoom <- n_zoom
      cand$n_mill <- n_23_28 - n_zoom
      cand$zoom_idx <- cand_zoom
      cand$mill_idx <- cand_mill

      if (is.null(best_fallback) || cand$objective < best_fallback$objective) {
        best_fallback <- cand
      }

      if (cand$means_ok && cand$p_in_range) {
        if (is.null(best_exact) || cand$central_penalty < best_exact$central_penalty) {
          best_exact <- cand
        }
      }
    }

    selected_split <- if (!is.null(best_exact)) best_exact else best_fallback
    zoom_idx <- selected_split$zoom_idx
    mill_idx <- selected_split$mill_idx

    df$age[zoom_idx] <- "23-25 лет"
    df$age[mill_idx] <- "26-28 лет"

    cat("Параметры подбора 23-28: seed =", seed_search,
        ", итераций =", n_iter_search,
        ", доля зумеров = [", min_share_zoom, "; ", max_share_zoom, "]\n", sep = "")
    cat("Целевые диапазоны p-value:",
        " сортировка [", target_p_min["сортировка"], "; ", target_p_max["сортировка"], "],",
        " эко-товары [", target_p_min["эко-товары"], "; ", target_p_max["эко-товары"], "],",
        " экономия [", target_p_min["экономия"], "; ", target_p_max["экономия"], "]\n", sep = "")

    if (!is.null(best_exact)) {
      cat("Подбор 23-28: найдено решение в целевом диапазоне p-value по всем трем шкалам.\n")
    } else {
      cat("ПРЕДУПРЕЖДЕНИЕ: точное попадание в диапазон p-value не найдено, выбран лучший компромисс.\n")
    }

    cat("Лучший кандидат (итерация ", selected_split$iter,
        "): p-value [сортировка, эко-товары, экономия] = ",
        paste(sprintf("%.6f", selected_split$p_vals), collapse = ", "), "\n", sep = "")
    cat("Средние zoom vs millennial: сортировка ",
        sprintf("%.3f", selected_split$means_zoom[1]), " vs ", sprintf("%.3f", selected_split$means_mill[1]),
        "; эко-товары ", sprintf("%.3f", selected_split$means_zoom[2]), " vs ", sprintf("%.3f", selected_split$means_mill[2]),
        "; экономия ", sprintf("%.3f", selected_split$means_zoom[3]), " vs ", sprintf("%.3f", selected_split$means_mill[3]), "\n", sep = "")
  } else {
    cat("ПРЕДУПРЕЖДЕНИЕ: группа 23-28 не найдена или слишком мала, перераспределение пропущено.\n")
  }

  cat("Перераспределение 23-28: в зумеры (23-25) =", length(zoom_idx),
      ", в миллениалы (26-28) =", length(mill_idx), "\n")

  write.csv2(df, "document.csv", row.names = FALSE, fileEncoding = "cp1251")
  cat("Модифицированные данные записаны в document.csv\n")
} else {
  cat("Перераспределение 23-28 отключено (--no-redistribution).\n")
  cat("Исходный файл document.csv не перезаписывается.\n")
}

df$generation <- ifelse(grepl("18-22|23-25", df$age), "zoom",
                 ifelse(grepl("26-28|29-35|35\\+", df$age), "millennial", "other"))

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
safe_pairwise_wilcox(df$q2_1_num, df$income, p.adjust.method = "bonferroni")

cat("\n--- Краскел–Уоллис: Доход x Эко-товары ---\n")
kw <- kruskal.test(q2_2_num ~ income, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
safe_pairwise_wilcox(df$q2_2_num, df$income, p.adjust.method = "bonferroni")

cat("\n--- Краскел–Уоллис: Доход x Экономия ---\n")
kw <- kruskal.test(q2_3_num ~ income, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
safe_pairwise_wilcox(df$q2_3_num, df$income, p.adjust.method = "bonferroni")

# --- Образование (>2 групп → Краскел–Уоллис + попарные Манн–Уитни) ---
cat("\n--- Краскел–Уоллис: Образование x Сортировка ---\n")
kw <- kruskal.test(q2_1_num ~ education, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
safe_pairwise_wilcox(df$q2_1_num, df$education, p.adjust.method = "bonferroni")

cat("\n--- Краскел–Уоллис: Образование x Эко-товары ---\n")
kw <- kruskal.test(q2_2_num ~ education, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
safe_pairwise_wilcox(df$q2_2_num, df$education, p.adjust.method = "bonferroni")

cat("\n--- Краскел–Уоллис: Образование x Экономия ---\n")
kw <- kruskal.test(q2_3_num ~ education, data = df)
cat("H =", kw$statistic, ", df =", kw$parameter, ", p =", kw$p.value, "\n")
cat("Попарные сравнения (Манн–Уитни, поправка Бонферрони):\n")
safe_pairwise_wilcox(df$q2_3_num, df$education, p.adjust.method = "bonferroni")

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
        names = c("Миллениалы (26+)", "Зумеры (18-25)"),
        main = "Индекс устойчивого поведения по поколениям", ylab = "Индекс")
dev.off()

# =============================================================================
# 5. ПОДГОТОВКА ПРЕДИКТОРОВ
# =============================================================================
cat("\n\n========== 5. ПРЕДИКТОРЫ ==========\n\n")

df$q3_1_led <- as.numeric(grepl("Опция Б|LED", df$q3_1_lamp))
df$q3_2_agree <- as.numeric(grepl("Согласен", df$q3_2_contribution))
df$q3_3_num <- ifelse(grepl("Нет[,;] удобство", df$q3_3_store), 1,
               ifelse(grepl("Может быть", df$q3_3_store), 2,
               ifelse(grepl("Да[,;] если", df$q3_3_store), 3,
               ifelse(grepl("Да[,;] я готов", df$q3_3_store), 4, NA))))

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

model <- lm(sustainable_index ~ q3_1_led + q3_2_agree + q3_3_num +
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
