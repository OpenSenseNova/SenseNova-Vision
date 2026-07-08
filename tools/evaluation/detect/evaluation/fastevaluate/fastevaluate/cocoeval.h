#ifndef _COCOEVAL_H_
#define _COCOEVAL_H_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <string.h>
#include <time.h>

#include <pthread.h>

#include "cJSON.h"
#include "maskApi.h"

using namespace std;

struct Params {
    int poolsize = 32;
    bool agnostic = false;
    string iouType = "bbox";
    vector<int> imgIds;
    vector<int> catIds;
    vector<string> catNames;
    vector<double> iouThrs = {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
    vector<double> recThrs;
    vector<int> maxDets = {1, 10, 100};
    vector<vector<double> > areaRng = {
        {0, 1e5*1e5},
        {0, 32*32},
        {32*32, 96*96},
        {96*96, 1e5*1e5}
    };
    vector<string>areaRngLbl = {"all", "small", "medium", "large"};
    vector<string>imgRngLbl = {"r", "c", "f"};
    int useCats = 1;
    Params() {
        for(int i = 0; i <= 100; i++) {
            recThrs.push_back(i * 0.01);
        }
    }
};

struct GtInstance {
    int id;
    int image_id;
    int category_id;
    double bbox[4];
    RLE *segmentation;
    double area;
    bool ignore;
    bool ignore_s;
    bool iscrowd;
    GtInstance() {}
    GtInstance(
        int _id, int _image_id, int _category_id,
        double _x, double _y, double _w, double _h,
        double _area, bool _ignore, bool _iscrowd,
        RLE *_segmentation=NULL
    ) {
        id = _id; image_id = _image_id; category_id = _category_id;
        bbox[0] = _x; bbox[1] = _y; bbox[2] = _w; bbox[3] = _h;
        area = _area; ignore = _ignore; iscrowd = _iscrowd;
        segmentation = _segmentation;
    }
};

struct DtInstance {
    int id;
    int image_id;
    int category_id;
    double bbox[4];
    RLE *segmentation;
    double area;
    double score;
    bool ignore;
    DtInstance() {}
    DtInstance(
        int _id, int _image_id, int _category_id,
        double _x, double _y, double _w, double _h,
        double _score, bool _ignore,
        RLE *_segmentation=NULL
    ) {
        id = _id; image_id = _image_id; category_id = _category_id;
        bbox[0] = _x; bbox[1] = _y; bbox[2] = _w; bbox[3] = _h;
        score = _score;
        area = bbox[2] * bbox[3];
        ignore = _ignore;
        segmentation = _segmentation;
    }

};

struct EvalResult {
    int image_id;
    int category_id;
    vector<double> aRng;
    int maxDet;
    vector<vector<int> > dtm;
    vector<vector<int> > gtm;
    vector<double> dtScores;
    vector<int> gtIg;
    vector<vector<int> > dtIg;
    bool empty;
    EvalResult() {empty = true;}
    EvalResult(
        int _image_id, int _category_id, vector<double>_aRng,
        int _maxDet, vector<vector<int> > _dtm, vector<vector<int> > _gtm,
        vector<double> _dtScores, vector<int> _gtIg, vector<vector<int> > _dtIg
    ) {
        image_id = _image_id; category_id = _category_id; aRng = _aRng;
        maxDet = _maxDet; dtm = _dtm; gtm = _gtm;
        dtScores = _dtScores; gtIg = _gtIg; dtIg = _dtIg;
        empty = false;
    }
};


bool cmp_score(const DtInstance &x1, const DtInstance &x2);

void merge_sort_dt(vector<DtInstance> &v, int l, int r);

bool cmp_gig(const pair<int, int> &a, const pair<int, int> &b);

void merge_sort_gig(vector<pair<int, int> > &v, int l, int r);

bool cmp_dscore(const pair<double, int> &a, const pair<double, int> &b);

void merge_sort_dscore(vector<pair<double, int> > &v, int l, int r);

string find_filename(string path);

string path_join(string prefix, string path);

class COCOEval {

    public:
        map<int, map<int, vector<GtInstance> > > _gts;
        map<int, map<int, vector<DtInstance> > > _dts;
        map<int, int> cat2cnt;
        double* precision;
        double* recall;
        float det_thre;

        map<string, int> catname2id;
        Params params;
        string _gt_path, _dt_path;
        bool _iou_type; // bbox: true; segm: false

        COCOEval();
        COCOEval(string gt_path, string dt_path, string type, int agnostic, float thre, string iou_type);
        ~COCOEval();
        void load_gts(string gt_path);
        void load_dts(string dt_path);
        void load_dts_tsv(string dt_path);
        void evaluate();
        static void* evaluateImgStatic(void *args);
        void evaluateImg(int k);
        double* computeIoU(int imgId, int catId);
        map<string, vector<double> > accumulate();
};

class LVISEval:public COCOEval {
    public:
        LVISEval(string gt_path, string dt_path, string type, float thre, string iou_type);
        map<int, set<int> > img_pl;
        map<int, set<int> > img_nl;
        map<int, set<int> > img_nel;
        map<string, vector<int> > freq_groups;
        void load_gts(string gt_path);
        void load_dts(string dt_path);
        void load_dts_tsv(string dt_path);
        map<string, vector<double> > accumulate();
};

#endif
