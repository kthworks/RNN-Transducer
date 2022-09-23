import argparse
from preprocess.grapheme import sentence_to_grapheme
from preprocess.preprocess import preprocess
from preprocess.character import generate_character_labels, generate_character_script
from preprocess.subword import train_sentencepiece, sentence_to_subwords


def _get_parser():
    ''' Get arguments parser '''
    parser = argparse.ArgumentParser(description='KsponSpeech Preprocess')
    parser.add_argument('--dataset_path', type=str,
                        default='E:/KsponSpeech/original',
                        help='path of original dataset')
    parser.add_argument('--vocab_dest', type=str,
                        default='E:/KsponSpeech',
                        help='destination to save character / subword labels file')
    parser.add_argument('--output_unit', type=str,
                        default='character',
                        help='character or subword or grapheme')
    parser.add_argument('--save_path', type=str,
                        default='./data',
                        help='path of data')
    parser.add_argument('--preprocess_mode', type=str,
                        default='phonetic',
                        help='Ex) (70%)/(칠 십 퍼센트) 확률이라니 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)/(백 프로)가 왜 안돼?'
                             'phonetic: 칠 십 퍼센트 확률이라니 모 몬 소리야 진짜 백 프로가 왜 안돼?'
                             'spelling: 70% 확률이라니 뭐 뭔 소리야 진짜 100%가 왜 안돼?')
    parser.add_argument('--vocab_size', type=int,
                        default=5000,
                        help='size of vocab (default: 5000)')

    return parser


def log_info(opt):
    print('Dataset Path : %s' % opt.dataset_path)
    print('Vocab Destination : %s' % opt.vocab_dest)
    print('Save Path : %s' % opt.save_path)
    print('Output-Unit : %s' % opt.output_unit)
    print('Preprocess Mode : %s' % opt.preprocess_mode)


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    log_info(opt)

    audio_paths, transcripts = preprocess(opt.dataset_path, opt.preprocess_mode)

    if opt.output_unit == 'character':
        generate_character_labels(transcripts, opt.vocab_dest)
        generate_character_script(audio_paths, transcripts, opt.vocab_dest, opt.save_path)

    elif opt.output_unit == 'subword':
        train_sentencepiece(transcripts, opt.save_path, opt.vocab_size)
        sentence_to_subwords(audio_paths, transcripts, opt.save_path)

    elif opt.output_unit == 'grapheme':
        sentence_to_grapheme(audio_paths, transcripts, opt.vocab_dest)

    else:
        raise ValueError('Unsupported preprocess method : {0}'.format(opt.output_unit))


if __name__ == '__main__':
    main()
