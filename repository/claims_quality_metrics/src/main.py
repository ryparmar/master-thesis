import argparse
import logging
import datetime
import os 
import sys

import utils


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--claims', type=str, default=None, help='Path to the csv with claims.')
    parser.add_argument('--labels', type=str, default=None, help='Path to the csv with labels.')
    parser.add_argument('--data', type=str, default=None, help='Path to the folder with jsonl files.')
    parser.add_argument('--cues', default='unigram', type=str, choices=['unigram', 'bigram', 'wordpiece', 'all'],
                                help='Valid representation of cues: unigram(s), bigram(s), wordpiece(s).')
    parser.add_argument('--cv', default=True, action='store_true', help='CrossValidation like sample 10 subdatasets')
    parser.add_argument('--no-cv', dest='cv', action='store_false', help='Compute directly using all the data.')
    parser.add_argument('--export', type=str, 
                                    help='Export all the results (not depends on topk argument) into a given folder.')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'dev', 'test', 'all'],
                                    help="Data split.")
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-cased', 
                                        help="Tokenizer used for getting wordpieces.")

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['cues'] = args_dict['cues'].lower().rstrip('s').strip() 

    print(args_dict, type(args.data))

    # Logging
    logging.basicConfig(filename='log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"Started at {start}")

    logger.info("GIVEN ARGUMENTS")
    for k, v in args_dict.items():
        logger.info(f"{k}: {v}")

    # Tokenizer
    tokenizer = None
    if args.cues in ['all', 'wordpieces', 'wordpiece']:
        try:
            import transformers
        except ImportError:
            print("Cannot import transformers library -- choose unigram/bigram cue representation instead.")
            sys.exit(0)
        tokenizer = transformers.BertTokenizerFast.from_pretrained(args.tokenizer)
        print(tokenizer)

    # Check arguments validity
    if args.claims and not os.path.exists(args.claims):
        logging.error(f"Given path to the claims file - {args.claims} - does not exist")
        sys.exit(0) 
    elif args.labels and not os.path.exists(args.labels):
        logging.error(f"Given path to the labels file - {args.labels} - does not exist")
        sys.exit(0)
    elif args.data and not os.path.exists(args.data):
        logging.error(f"Given path to the data foler - {args.data} - does not exist")
        sys.exit(0)
    if args.cues.lower().strip() not in ['unigram', 'unigrams', 'bigram', 'bigrams', 'wordpiece', 'wordpieces']:
        logging.error(f"Given cue representation - {args.cues} - is not valid.", 
                      f"Valid representation is one of the: ",
                      f"['unigram', 'unigrams', 'bigram', 'bigrams', 'wordpiece', 'wordpieces']")
        parser.print_help()
        sys.exit(0)
    
    # TODO enable to work with sql dump file as well
    logging.info(f'Processing input files as a single dataset...')
    data = utils.process_jsonl_in_folder(args.data, args.split) if args.data else utils.process_csv(args.claims, args.labels)
    NUM_LABELS = len(data.label.unique())  # num of possible labels (3 for FEVER dataset - supports, refutes, nei)
    print(f"Number of claims: {NUM_LABELS}")

    if args.cv:
        logging.info(f'Processing csv input files as a dataset of k samples...')
        data_k_folds = utils.get_k_fold(data, k=10)  # k = number of folds
        logging.info(f'Computing metrics into the result dataframe...')
        res = utils.get_result_frame(data_k_folds, args.cues, args.cv, NUM_LABELS, tokenizer)
    else:
        logging.info(f'Computing metrics into the result dataframe...')
        res = utils.get_result_frame(data, args.cues, args.cv, NUM_LABELS, tokenizer)

    print(f"\nTOP 20 CUES WITH A HIGHEST POTENTIAL TO CREATE A PATTERN\n{res[:20]}")
    if args.export:
        path = os.path.join(args.export, f"{args.cues}-util-cov.csv")
        logging.info(f'Saving into file {path}')
        res.to_csv(path, encoding="utf-8", index=True)

        with open(os.path.join(args.export, f"{args.cues}-util-cov-latex"), 'w') as fw:
            fw.write(res[:20].to_latex(index=True))


    # Dataset-weighted Cue Information (DCI)
    skip = 4
    dci = utils.calculate_dci(data, args.cues, skip, 3, tokenizer)
    print(f"\nTOP 20 CUES PROVIDING HIGHEST INFORMATION GAIN\n{dci[:20]}")
    if args.export:
        s = f"{skip}-skip-" if args.cues in ['bigram', "trigram"] else ""
        path = os.path.join(args.export, f"{s}{args.cues}-dci.csv")
        logging.info(f'Saving into file {path}')
        dci.to_csv(path, encoding="utf-8", index=True)

        with open(os.path.join(args.export, f"{args.cues}-dci-latex"), 'w') as fw:
            fw.write(dci[:20].to_latex(index=False))



    
