
# Extraction of a ten minutes data interval from the alibaba cluster data
# in the alibaba_clusterdata_v2018 folder

# caution: takes about 1 hour to extract

# uncomment if directory is not already present
#mkdir alibaba_short_time_interval_test_data

# Extracting a time interval from container_usage
## working code for cutting over time interval

awk -F "\"*,\"*" '{ if ($1 >= 386600 && $1 <= 386615)  print }' container_usage.csv > container_usage_chunk.csv

## put in a header
sed '1itimestamp,container_id,machine_id,cpu,mem' container_usage_chunk.csv > container_usage_chunk_header.csv

mv container_usage_chunk_header.csv alibaba_short_time_interval_test_data/container_usage.csv

####################################################
# for node_usage

awk -F "\"*,\"*" '{ if ($1 >= 386605 && $1 <= 386615) print}' node_usage.csv > node_usage_chunk.csv

## put the header

sed '1itimestamp,machine_id,cpu,mem' node_usage_chunk.csv > node_usage_chunk_header.csv


mv node_usage_chunk_header.csv alibaba_short_time_interval_test_data/node_usage.csv

#####################################################
