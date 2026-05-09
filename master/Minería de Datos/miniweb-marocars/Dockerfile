FROM rocker/r-ver:4.5.2

RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libicu-dev \
    libgit2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('plumber','readxl','jsonlite','randomForest','caret'), repos='https://cloud.r-project.org')"

WORKDIR /app

COPY api /app/api
COPY data /app/data

WORKDIR /app/api

EXPOSE 10000

CMD ["Rscript", "start-api.R"]
