library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# Assigned path to the Markdown file used for the plot. The folder and file will be created using "make benchmark".
file_path <- "benchmarks/benchmark_results.md"

# Read the Markdown table as data, treating all rows as data, no headers.
df <- read_delim(file_path, delim = "|", col_names = FALSE, skip = 1, trim_ws = TRUE, show_col_types = FALSE)

# Manually assign the correct column names.
colnames(df) <- c("Empty1", "Command", "Mean_s", "Min_s", "Max_s", "Relative", "Empty2")

# Remove the first row, which contains non-data information.
df <- df %>% slice(-1)

# Remove empty columns (first and last).
df <- df %>%
  select(-Empty1, -Empty2) %>%
  filter(!is.na(Command))

# Clean the 'Mean_s', 'Min_s', 'Max_s' columns by removing ellipsis and ±. Also trimming the extra content.
df <- df %>%
  mutate(
    Mean_s = as.numeric(trimws(sub(" ±.*", "", Mean_s))),  
    Min_s = as.numeric(trimws(Min_s)),                     
    Max_s = as.numeric(trimws(Max_s))                      
  )

# Remove rows with NA values in key columns.
df <- df %>% filter(!is.na(Mean_s) & !is.na(Min_s) & !is.na(Max_s))

# Update the Command column based on the -m flag
df <- df %>%
  mutate(Command = case_when(
    grepl("-m1", Command) ~ "sequential",
    grepl("-m2", Command) ~ "parallel",
    grepl("-m3", Command) ~ "simd",
    TRUE ~ "unknown"  
  ))


# Setting the order of each column as the benchmark test provided the results.
df$Command <- factor(df$Command, levels = c("sequential", "parallel", "simd"))

# Print the used data in the terminal for diagnostic purposes.
print("Data in wide format for plotting:")
print(df)

# At this point it's possible to adjust the coloring for each column separately by using the specific HEX-code.
mean_color <- "#040403"  
min_color <- "#8EB897"   
max_color <- "#5B7553"   

# Extract system information
cpu_info <- system("lscpu | grep 'Model name' | awk -F: '{print $2}'", intern = TRUE)
cpu_cores <- system("lscpu | grep '^Core(s) per socket:' | awk -F: '{print $2}'", intern = TRUE)
cpu_threads <- system("lscpu | grep 'Thread(s) per core' | awk -F: '{print $2}'", intern = TRUE)
# cpu_speed <- system("lscpu | grep 'MHz' | awk -F: '{print $2}'", intern = TRUE)
# gpu_info <- system("lspci | grep -i 'vga\\|3d\\|2d'", intern = TRUE)
# ram_info <- system("free -h | grep 'Mem:' | awk '{print $2}'", intern = TRUE)

# Format the information
cpu_info <- trimws(cpu_info)
cpu_cores <- trimws(cpu_cores)
cpu_threads <- trimws(cpu_threads)
# cpu_speed <- trimws(cpu_speed)
# gpu_info <- trimws(gpu_info)
# ram_info <- trimws(ram_info)

# Create a formatted string for display
system_info <- paste0("CPU: ", cpu_info, "\n",
                      "Core(s): ", cpu_cores, ", Thread(s) per core: ", cpu_threads)
                     # , ", Speed: ", cpu_speed, " MHz\n",
                     # "GPU: ", gpu_info, "\n",
                     # "RAM: ", ram_info)

# Create the plot and add system information in the caption
p <- ggplot(df, aes(x = Command)) +
  geom_bar(aes(y = Mean_s, fill = "Mean"), stat = "identity", alpha = 0.8, width = 0.6) +
  geom_bar(aes(y = Min_s, fill = "Min"), stat = "identity", alpha = 0.9, width = 0.05, position = position_nudge(x = 0.375)) +
  geom_bar(aes(y = Max_s, fill = "Max"), stat = "identity", alpha = 0.9, width = 0.05, position = position_nudge(x = 0.325)) +

# Settings for the legend
  scale_fill_manual(
    name = "Legend",
    values = c("Mean" = mean_color, "Min" = min_color, "Max" = max_color),
    labels = c("Mean" = "Mean", "Min" = "Min", "Max" = "Max")
  ) +
  labs(
    title = "Benchmark Results",
    y = "Execution Time (s)",
    caption = paste0("System Information:\n", system_info)
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.title.x = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.background = element_rect(fill = "white", color = "white"),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 10),
    plot.caption = element_text(hjust = 0, face = "italic", size = 8)  # Adjust caption formatting
  )

# Save the plot as .png
ggsave("benchmarks/benchmark_plot.png", plot = p, width = 8, height = 6)
print("Plot saved successfully with system information.")
