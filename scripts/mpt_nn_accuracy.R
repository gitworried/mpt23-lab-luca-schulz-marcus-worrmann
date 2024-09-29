# For some reasons there where plenty of dlpyr message for every plot. Since they don't harm in any way following code ensure to suppress any dplyr startup messages.
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))

extract_accuracy_data <- function(file_path) {
  lines <- readLines(file_path)  # Read the markdown file

  # Initialize empty lists for data storaging.
  epoch <- c()
  mode <- c()
  accuracy <- c()

  pattern <- "Mode: (sequential|parallel|SIMD) \\| Epoch ([0-9]+)/10.*Accuracy: ([0-9]+\\.[0-9]+)%"

  # Loop for every line to search and find possible matches.
  for (line in lines) {
    match <- regmatches(line, regexec(pattern, line))
    if (length(match[[1]]) > 1) {
      mode <- c(mode, match[[1]][2])  # Mode (sequential, parallel, SIMD)
      epoch <- c(epoch, as.numeric(match[[1]][3]))  # Epoch number
      accuracy <- c(accuracy, as.numeric(match[[1]][4]))  # Accuracy value
    }
  }

  # Create a data frame with the extracted data
  data <- data.frame(Epoch = epoch, Mode = mode, Accuracy = accuracy)

  # Factorize the Mode column. This will ensure ggplot2 can interpret it correctly
  data$Mode <- factor(data$Mode, levels = c("sequential", "parallel", "SIMD"))
  
  return(data)
}

# Extract the data from the markdown file
data <- extract_accuracy_data('benchmarks/accuracy_results.md')

# Calculate the average per epoch for close up summary of all gathered data.
summary_data <- data %>%
  group_by(Epoch, Mode) %>%
  summarise(Average_Accuracy = mean(Accuracy), .groups = 'drop')

# At this point it's possible to adjust the coloring for each mode separately by using the specific HEX-code.
custom_colors <- c(
  "sequential" = "#611C35",
  "parallel"   = "#C84C09", 
  "SIMD"       = "#00AF54"
)

# Settings for the legend
p <- ggplot(summary_data, aes(x = Epoch, y = Average_Accuracy, color = Mode)) +
  geom_point(size = 4, alpha = 0.7) +  
  geom_line(linewidth = 0.55) +  
  labs(title = "Average Accuracy Results", x = "Epoch", y = "Average Accuracy (%)", color = "Legend:") +  
  scale_color_manual(values = custom_colors) +  
  scale_x_continuous(breaks = c(2, 4, 6, 8, 10)) + 
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white"),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.title = element_text(size = 14),
    legend.box.background = element_blank(),
    legend.box.margin = margin(0, 0, 0, 0)
  )

# Save the plot as .png
ggsave("benchmarks/accuracy_plot.png", plot = p, width = 10, height = 6)
print("Plot saved successfully with system information.")

