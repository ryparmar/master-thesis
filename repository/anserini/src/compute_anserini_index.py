# import os
# import logging
# import numpy as np
# import sys

# from utils import convert_collection_to_json
# from utils import 

# logging.basicConfig(filename='logs/anserini-run.log',
# 					format='%(asctime)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO)




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Converts FEVER jsonl wikipedia dump to anserini jsonl files.')
#     parser.add_argument('--collection_folder', required=True, help='FEVER wiki-pages directory.')
#     parser.add_argument('--output_folder', required=True, help='Output directory.')
#     parser.add_argument('--max_docs_per_file',
#                         default=1000000,
#                         type=int,
#                         help='Maximum number of documents in each jsonl file.')
#     parser.add_argument('--granularity',
#                         required=True,
#                         choices=['paragraph', 'sentence'],
#                         help='The granularity of the source documents to index. Either "paragraph" or "sentence".')
#     args = parser.parse_args()

#     if not os.path.exists(args.output_folder):
#         os.makedirs(args.output_folder)

#     convert_collection(args)

#     print('Done!')