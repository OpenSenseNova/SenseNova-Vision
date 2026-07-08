from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.eval import COCOEvalCap as _COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


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
