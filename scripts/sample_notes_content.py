"""
Original study-note content authored for the DataMind RAG Assistant project.

This module holds plainly-written, original educational notes covering
topics that don't currently have an official, freely downloadable PDF
(e.g. current scikit-learn / SQL / statistics / EDA / AI fundamentals
documentation is HTML-only, paywalled, or otherwise not redistributable).

Each entry becomes one PDF document in data/documents/, complete with
section headings so page-level citations remain meaningful. Content here
is original writing for this project — not copied from any book, article,
or third-party documentation.
"""

from __future__ import annotations

# Each document is a list of (heading, paragraphs) sections.
# `paragraphs` is a list of plain strings (one per paragraph).

DOCUMENTS: dict[str, dict] = {
    "sql_for_data_analysis_notes.pdf": {
        "title": "SQL for Data Analysis - Study Notes",
        "sections": [
            (
                "What is SQL?",
                [
                    "SQL (Structured Query Language) is the standard language used to define, "
                    "query, and manipulate data stored in relational databases. A relational "
                    "database organizes data into tables made up of rows and columns, where "
                    "each row represents a single record and each column represents an "
                    "attribute of that record.",
                    "Analysts use SQL to pull exactly the data they need out of large "
                    "databases, filter and aggregate it, and combine information that is "
                    "spread across multiple tables, before moving it into tools like Pandas, "
                    "Excel, or a BI dashboard for further analysis.",
                ],
            ),
            (
                "SELECT, WHERE, and ORDER BY",
                [
                    "The SELECT statement is the starting point of almost every SQL query. It "
                    "specifies which columns to retrieve from a table, for example "
                    "'SELECT customer_id, order_date FROM orders'. Using SELECT * retrieves "
                    "every column, which is convenient for exploration but often wasteful in "
                    "production queries.",
                    "The WHERE clause filters rows based on a condition, such as "
                    "'WHERE order_date >= 2024-01-01'. Conditions can be combined with AND, "
                    "OR, and NOT, and can use comparison operators, BETWEEN, IN, and LIKE for "
                    "pattern matching on text.",
                    "ORDER BY sorts the result set by one or more columns, ascending by "
                    "default or descending when the DESC keyword is added. LIMIT (or TOP, "
                    "depending on the database engine) restricts how many rows are returned, "
                    "which is useful when previewing large tables.",
                ],
            ),
            (
                "Aggregation: GROUP BY and aggregate functions",
                [
                    "Aggregate functions such as COUNT, SUM, AVG, MIN, and MAX summarize many "
                    "rows into a single value. On their own they summarize the whole table; "
                    "combined with GROUP BY, they compute one summary value per group.",
                    "For example, 'SELECT region, SUM(revenue) FROM sales GROUP BY region' "
                    "returns total revenue for each region. The HAVING clause filters groups "
                    "after aggregation (for example, only regions with total revenue above a "
                    "threshold), which is different from WHERE, which filters individual rows "
                    "before aggregation.",
                ],
            ),
            (
                "What is a SQL JOIN?",
                [
                    "A JOIN combines rows from two or more tables based on a related column "
                    "between them, typically a key such as customer_id. Without joins, data "
                    "that is normalized across multiple tables (customers, orders, products) "
                    "could not be analyzed together in a single query.",
                    "An INNER JOIN returns only rows that have matching values in both "
                    "tables. A LEFT JOIN returns all rows from the left table and the "
                    "matching rows from the right table, filling in NULLs where there is no "
                    "match. A RIGHT JOIN does the mirror image, and a FULL OUTER JOIN returns "
                    "all rows from both tables, matched where possible.",
                    "Choosing the right join type matters for correctness: an INNER JOIN can "
                    "silently drop rows that don't have a match, which is a common source of "
                    "undercounting in analytical queries.",
                ],
            ),
            (
                "Common analytical patterns",
                [
                    "Window functions (e.g. ROW_NUMBER, RANK, SUM(...) OVER (PARTITION BY "
                    "...)) let an analyst compute running totals, rankings, or per-group "
                    "calculations without collapsing rows the way GROUP BY does, which is "
                    "especially useful for time-series and leaderboard-style analysis.",
                    "Common Table Expressions (CTEs), written with the WITH keyword, let you "
                    "break a complex query into named, readable steps, which is often "
                    "preferred over deeply nested subqueries when preparing data for a "
                    "downstream Pandas or visualization step.",
                ],
            ),
        ],
    },
    "statistics_fundamentals_notes.pdf": {
        "title": "Statistics Fundamentals for Data Analysis - Study Notes",
        "sections": [
            (
                "Descriptive statistics",
                [
                    "Descriptive statistics summarize the main features of a dataset. The "
                    "mean (average), median (middle value), and mode (most frequent value) "
                    "describe the center of a distribution, while the range, variance, and "
                    "standard deviation describe how spread out the values are.",
                    "The median is more robust to outliers than the mean: a handful of "
                    "extreme values can pull the mean far away from where most of the data "
                    "actually sits, while the median stays near the bulk of the observations.",
                ],
            ),
            (
                "Distributions and the normal distribution",
                [
                    "A distribution describes how often each value (or range of values) "
                    "occurs in a dataset. The normal (Gaussian) distribution is a common, "
                    "symmetric, bell-shaped distribution that appears frequently in natural "
                    "and social phenomena, and underlies many classical statistical tests.",
                    "Skewness describes asymmetry in a distribution: a right-skewed "
                    "(positively skewed) distribution has a long tail toward higher values, "
                    "which is common for variables like income or house prices.",
                ],
            ),
            (
                "Correlation vs. causation",
                [
                    "Correlation measures how strongly two variables move together, usually "
                    "summarized with a correlation coefficient between -1 and 1. A value near "
                    "1 means the variables tend to increase together, near -1 means one "
                    "tends to increase as the other decreases, and near 0 means little linear "
                    "relationship.",
                    "Correlation does not imply causation: two variables can be correlated "
                    "because one causes the other, because a third variable causes both, or "
                    "simply by coincidence. Establishing causation generally requires "
                    "controlled experiments or careful causal-inference methods, not just an "
                    "observed correlation.",
                ],
            ),
            (
                "Hypothesis testing basics",
                [
                    "A hypothesis test evaluates whether an observed effect in sample data is "
                    "likely to reflect a real effect in the broader population, or could "
                    "plausibly be due to random chance. The null hypothesis typically "
                    "represents 'no effect', and the test produces a p-value: the probability "
                    "of seeing a result at least as extreme as the observed one if the null "
                    "hypothesis were true.",
                    "A small p-value (commonly below 0.05) is often treated as evidence "
                    "against the null hypothesis, but a p-value threshold is a convention, "
                    "not a guarantee of practical significance — a statistically significant "
                    "effect can still be too small to matter in practice.",
                ],
            ),
        ],
    },
    "exploratory_data_analysis_notes.pdf": {
        "title": "Exploratory Data Analysis (EDA) - Study Notes",
        "sections": [
            (
                "What is exploratory data analysis?",
                [
                    "Exploratory data analysis (EDA) is the process of investigating a "
                    "dataset — before formal modeling — to understand its structure, spot "
                    "patterns, detect anomalies, and check underlying assumptions, using a "
                    "combination of summary statistics and visualizations.",
                    "EDA typically comes right after data collection and cleaning, and its "
                    "findings shape everything downstream: which features are worth "
                    "engineering, which model family might fit the data, and which "
                    "data-quality issues need to be fixed first.",
                ],
            ),
            (
                "Why data cleaning matters",
                [
                    "Real-world data is rarely ready to analyze as-is: it commonly contains "
                    "missing values, duplicate records, inconsistent formatting (like mixed "
                    "date formats or inconsistent capitalization), and outliers caused by "
                    "measurement or data-entry errors.",
                    "Skipping data cleaning risks building an analysis or model on flawed "
                    "inputs — the well-known idea of 'garbage in, garbage out'. Careful "
                    "cleaning (handling missing values deliberately, standardizing formats, "
                    "removing or investigating duplicates) makes every later step more "
                    "trustworthy.",
                ],
            ),
            (
                "A typical EDA workflow",
                [
                    "A common workflow starts with basic inspection: checking the shape of "
                    "the data, column data types, and the first few rows, followed by "
                    "summary statistics (counts, means, min/max, missing-value counts) for "
                    "each column.",
                    "Next comes univariate analysis — looking at one variable at a time with "
                    "histograms, box plots, or bar charts — followed by bivariate or "
                    "multivariate analysis, such as scatter plots or correlation heatmaps, to "
                    "see how variables relate to each other and to any target variable of "
                    "interest.",
                    "Throughout, the analyst asks questions of the data: Are there missing "
                    "values, and are they random or systematic? Are there outliers, and are "
                    "they errors or genuine extreme cases? Do relationships between variables "
                    "match domain expectations?",
                ],
            ),
            (
                "The role of visualization",
                [
                    "Visualization is central to EDA because summary statistics alone can "
                    "hide important structure — famously, very different datasets can share "
                    "nearly identical means, variances, and correlations while looking "
                    "completely different when plotted (a phenomenon popularized by "
                    "'Anscombe's quartet').",
                    "Libraries like Matplotlib and Seaborn are commonly used for this stage "
                    "in Python: Matplotlib provides fine-grained control over plots, while "
                    "Seaborn builds on top of Matplotlib to make common statistical "
                    "visualizations (distribution plots, box plots, pair plots, heatmaps) "
                    "faster to produce with good default styling.",
                ],
            ),
        ],
    },
    "machine_learning_fundamentals_notes.pdf": {
        "title": "Machine Learning Fundamentals - Study Notes",
        "sections": [
            (
                "What is machine learning?",
                [
                    "Machine learning is a set of techniques that let computer programs "
                    "improve their performance on a task by learning patterns from data, "
                    "rather than being explicitly programmed with fixed rules for every "
                    "situation.",
                    "It is commonly split into supervised learning (learning from labeled "
                    "examples), unsupervised learning (finding structure in unlabeled data), "
                    "and reinforcement learning (learning by trial and error through rewards "
                    "and penalties).",
                ],
            ),
            (
                "Supervised learning: regression and classification",
                [
                    "In supervised learning, a model is trained on input-output pairs so it "
                    "can predict the output for new, unseen inputs. Regression tasks predict "
                    "a continuous number, such as a house price, while classification tasks "
                    "predict a discrete category, such as whether an email is spam.",
                    "Common supervised algorithms include linear and logistic regression, "
                    "decision trees, random forests, gradient-boosted trees, and neural "
                    "networks — each with different trade-offs in interpretability, training "
                    "speed, and predictive performance.",
                ],
            ),
            (
                "Feature engineering",
                [
                    "Feature engineering is the process of transforming raw data into "
                    "informative input variables (features) that make it easier for a model "
                    "to find useful patterns. This can include creating new columns (like "
                    "extracting the day-of-week from a date), encoding categorical variables "
                    "as numbers, and scaling numeric features to comparable ranges.",
                    "Good feature engineering often has a bigger impact on model performance "
                    "than the choice of algorithm itself, because it directly shapes what "
                    "information the model has access to.",
                ],
            ),
            (
                "Train/test split and overfitting",
                [
                    "A train/test split divides the available data into a portion used to "
                    "fit the model (the training set) and a portion held back purely for "
                    "evaluation (the test set), so that performance is measured on data the "
                    "model has never seen during training.",
                    "Overfitting happens when a model learns the noise and idiosyncrasies of "
                    "the training data too closely, achieving excellent training performance "
                    "but generalizing poorly to new data. Underfitting is the opposite "
                    "problem: the model is too simple to capture the real patterns, so it "
                    "performs poorly on both training and test data.",
                    "Cross-validation extends this idea by splitting the data into several "
                    "folds and rotating which fold is held out for testing, giving a more "
                    "reliable estimate of how a model will perform on new data than a single "
                    "train/test split.",
                ],
            ),
            (
                "Model evaluation",
                [
                    "For regression, common evaluation metrics include mean absolute error "
                    "(MAE), mean squared error (MSE), and R-squared, which measures how much "
                    "of the variance in the target the model explains.",
                    "For classification, common metrics include accuracy, precision, recall, "
                    "and F1-score. Accuracy alone can be misleading on imbalanced datasets "
                    "(for example, if 95% of emails are not spam, a model that always "
                    "predicts 'not spam' gets 95% accuracy while being useless), which is why "
                    "precision and recall are often reported alongside it.",
                ],
            ),
        ],
    },
    "ai_and_generative_ai_fundamentals_notes.pdf": {
        "title": "AI and Generative AI Fundamentals - Study Notes",
        "sections": [
            (
                "What is artificial intelligence?",
                [
                    "Artificial intelligence (AI) broadly refers to computer systems that "
                    "perform tasks which typically require human intelligence, such as "
                    "understanding language, recognizing images, or making decisions under "
                    "uncertainty. Machine learning is the dominant modern approach to "
                    "building AI systems, using data-driven pattern learning rather than "
                    "hand-coded rules.",
                    "Deep learning is a subfield of machine learning built on neural networks "
                    "with many layers, which has driven much of the recent progress in "
                    "computer vision, speech recognition, and natural language processing.",
                ],
            ),
            (
                "What is generative AI?",
                [
                    "Generative AI refers to models that can produce new content — text, "
                    "images, audio, or code — rather than simply classifying or predicting a "
                    "single number or label. Large language models (LLMs) are a prominent "
                    "example: they are trained on huge amounts of text to predict the next "
                    "word in a sequence, which lets them generate coherent, contextually "
                    "relevant text.",
                    "Other generative model families include diffusion models, widely used "
                    "for image generation, and generative adversarial networks (GANs), an "
                    "earlier architecture where two networks — a generator and a "
                    "discriminator — are trained together in competition.",
                ],
            ),
            (
                "Retrieval-augmented generation (RAG)",
                [
                    "Retrieval-augmented generation combines a language model with an "
                    "external knowledge source: instead of relying purely on what the model "
                    "memorized during training, the system first retrieves relevant "
                    "documents or passages from a knowledge base, then asks the model to "
                    "answer using that retrieved context.",
                    "This approach helps ground answers in verifiable, up-to-date source "
                    "material, reduces the chance of the model inventing plausible-sounding "
                    "but incorrect facts (a failure mode often called 'hallucination'), and "
                    "makes it possible to cite the specific source and page an answer came "
                    "from — which is exactly the architecture this project (DataMind) "
                    "implements.",
                ],
            ),
            (
                "Embeddings and vector search",
                [
                    "An embedding is a numeric vector representation of a piece of text (or "
                    "image, or other data) such that semantically similar inputs end up with "
                    "similar vectors. Embedding models are typically trained so that the "
                    "distance between two vectors reflects how related their meanings are, "
                    "not just whether they share exact words.",
                    "A vector database stores these embeddings and lets you efficiently find "
                    "the vectors most similar to a given query vector — this is the "
                    "'retrieval' half of retrieval-augmented generation, and is what allows a "
                    "system to find the most relevant document chunks for a user's question "
                    "out of a large corpus.",
                ],
            ),
        ],
    },
}
