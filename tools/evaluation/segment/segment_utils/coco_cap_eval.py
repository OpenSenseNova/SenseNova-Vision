from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.eval import COCOEvalCap as _COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


MAX_PREDICTION_TOKENS = 2000


def truncate_prediction_tokens(captions, max_tokens=MAX_PREDICTION_TOKENS):
    """Limit tokenized prediction captions before metric computation."""
    truncated = {}
    truncated_count = 0
    for image_id, image_captions in captions.items():
        truncated[image_id] = []
        for caption in image_captions:
            tokens = caption.split()
            if len(tokens) > max_tokens:
                caption = " ".join(tokens[:max_tokens])
                truncated_count += 1
            truncated[image_id].append(caption)
    return truncated, truncated_count


class COCOEvalCap(_COCOEvalCap):
    def evaluate(self):
        imgIds = self.params["image_id"]
        gts = {}
        res = {}
        for imgId in imgIds:
            gt_anns = self.coco.imgToAnns[imgId]
            re_anns = self.cocoRes.imgToAnns[imgId]
            if len(re_anns) != 1:
                print(f"imgId = {imgId}, len(re_anns) = {len(re_anns)}")
                continue
            gts[imgId] = gt_anns
            res[imgId] = re_anns

        # =================================================
        # Set up scorers
        # =================================================
        print("tokenization...")
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        res, truncated_count = truncate_prediction_tokens(res)
        if truncated_count:
            print(
                f"truncated {truncated_count} prediction caption(s) to "
                f"{MAX_PREDICTION_TOKENS} tokens for caption metrics"
            )

        # =================================================
        # Set up scorers
        # =================================================
        print("setting up scorers...")
        scorers = [
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print("computing %s score..." % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f" % (method, score))
        self.setEvalImgs()
