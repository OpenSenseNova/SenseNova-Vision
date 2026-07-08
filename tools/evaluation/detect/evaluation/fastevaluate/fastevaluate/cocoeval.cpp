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
#include <sys/stat.h>
#include <sys/types.h>

#include <pthread.h>

#include "cJSON.h"
#include "maskApi.h"
#include "cocoeval.h"

using namespace std;

bool cmp_score(const DtInstance &x1, const DtInstance &x2) {
    return x1.score > x2.score;
}

void merge_sort_dt(vector<DtInstance> &v, int l, int r) {
    if (l < r) {
        int mid = (l + r) / 2;
        merge_sort_dt(v, l, mid);
        merge_sort_dt(v, mid+1, r);
        inplace_merge(v.begin() + l, v.begin() + mid + 1, v.begin() + r + 1, cmp_score);
    }
}

bool cmp_gig(const pair<int, int> &a, const pair<int, int> &b) {
    return a.first < b.first;
}

void merge_sort_gig(vector<pair<int, int> > &v, int l, int r) {
    if (l < r) {
        int mid = (l + r) / 2;
        merge_sort_gig(v, l, mid);
        merge_sort_gig(v, mid+1, r);
        inplace_merge(v.begin() + l, v.begin() + mid + 1, v.begin() + r + 1, cmp_gig);
    }
}

bool cmp_dscore(const pair<double, int> &a, const pair<double, int> &b) {
    return a.first > b.first;
}

void merge_sort_dscore(vector<pair<double, int> > &v, int l, int r) {
    if (l < r) {
        int mid = (l + r) / 2;
        merge_sort_dscore(v, l, mid);
        merge_sort_dscore(v, mid+1, r);
        inplace_merge(v.begin() + l, v.begin() + mid + 1, v.begin() + r + 1, cmp_dscore);
    }
}

string find_filename(string path) {
    int len = path.size();
    int pos = 0;
    char tmp[1000], ptr=0;
    for (int i = 0; i < len; i++) {
        if (path[i] == '/') {
            pos = i; 
        }
    }
    for (int i = pos+1; i < len; i++) {
        tmp[ptr++] = path[i];
    }
    tmp[ptr++] = '\0';
    return string(tmp);
}

vector<string> split(string s, char p) {
    vector<string> res;
    int len = s.size();
    int pos = 0;
    char tmp[1000], ptr=0;
    for (int i = 0; i < len + 1; i++) {
        if (s[i] == p || s[i] == '\0') {
            tmp[ptr++] = '\0';
            res.push_back(string(tmp)); 
            ptr = 0;
        }
        else {
            tmp[ptr++] = s[i];
        }
    } 
    return res;
}


string path_join(string prefix, string path) {
    vector<string> tmp = split(prefix, '/');
    string folder = ".";
    for (int i = 0; i < tmp.size(); i++) {
        folder = folder + '/' + tmp[i];
        mkdir(folder.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    }
    return prefix + "/" + path;
}


struct threadp {
    COCOEval *p;
    int k;
    threadp(){}
    threadp(COCOEval *_p, int _k) {
        p = _p;
        k = _k;
    }
};


COCOEval::COCOEval() {} 

COCOEval::COCOEval(string gt_path, string dt_path, string type, int agnostic, float thre, string iou_type) {
    params.agnostic = agnostic;
    _iou_type = iou_type == "segm";
    det_thre = thre;
    clock_t start, end;
    double dur;
    start = clock();
    load_gts(gt_path);
    end = clock();
    dur = (double)(end - start)/CLOCKS_PER_SEC;
    printf("load gts cost %lf\n", dur);


    start = clock();

    if (type == "json") {
        load_dts(dt_path);
    }
    else if (type == "tsv") {
        load_dts_tsv(dt_path);
    }
    printf("load dt\n");

    end = clock();
    dur = (double)(end - start)/CLOCKS_PER_SEC;
    printf("load dts cost %lf\n", dur);

}

COCOEval::~COCOEval() {}

void COCOEval::load_gts(string gt_path) {
    _gt_path = gt_path;
    ifstream gt_file(gt_path);
    string gt_json((istreambuf_iterator<char>(gt_file)),
                   istreambuf_iterator<char>());

    cJSON* gt_root = cJSON_Parse(gt_json.c_str());
    cJSON* gt_images = cJSON_GetObjectItem(gt_root, "images");
    cJSON* img_item = gt_images->child;

    map<int, pair<int, int> > imgid2hw;

    while (img_item != NULL) {
        cJSON* id = cJSON_GetObjectItem(img_item, "id");
        cJSON* width = cJSON_GetObjectItem(img_item, "width");
        cJSON* height = cJSON_GetObjectItem(img_item, "height");
        imgid2hw[id->valueint] = make_pair(height->valueint, width->valueint);
        params.imgIds.push_back(id->valueint);
        img_item = img_item->next;
    }


    cJSON* gt_annotations = cJSON_GetObjectItem(gt_root, "annotations");
    cJSON* ann_item = gt_annotations->child;
    while (ann_item != NULL) {
        cJSON* id = cJSON_GetObjectItem(ann_item, "id");
        cJSON* image_id = cJSON_GetObjectItem(ann_item, "image_id");
        cJSON* category_id = cJSON_GetObjectItem(ann_item, "category_id");
        cJSON* bbox = cJSON_GetObjectItem(ann_item, "bbox");
        cJSON* iscrowd = cJSON_GetObjectItem(ann_item, "iscrowd");
        cJSON* area = cJSON_GetObjectItem(ann_item, "area");
        cJSON* x = bbox->child;
        cJSON* y = x->next;
        cJSON* w = y->next;
        cJSON* h = w->next;

        int imgId = image_id->valueint;
        int catId = category_id->valueint;

        int isCrowd = 0;
        if (iscrowd != NULL) {
            isCrowd = iscrowd->valueint;
        }

        if (params.agnostic) {
            catId = 1;
        }
        if (_iou_type) {
            cJSON* segmentation = cJSON_GetObjectItem(ann_item, "segmentation");
            pair<int, int> hw = imgid2hw[imgId];

            RLE *MR = new RLE();
            cJSON* counts = cJSON_GetObjectItem(segmentation, "counts");
            if (counts != NULL) {
                if (cJSON_IsArray(counts)) {
                    vector<uint> cnts;
                    cJSON *counts_item = counts->child;
                    while (counts_item != NULL) {
                        cnts.push_back(counts_item->valueint);
                        counts_item = counts_item->next;
                    }
                    rleInit(MR, hw.first, hw.second, cnts.size(), cnts.data());
                }
                else {
                    string rle = counts->valuestring; 
                    rleFrString(MR, rle.c_str(), hw.first, hw.second);
                }
            }
            else {
                cJSON* polys = segmentation->child;
                vector<RLE> rles;
                while (polys != NULL) {
                    cJSON* poly_item = polys->child;
                    vector<double> poly_data;
                    while (poly_item != NULL) {
                        poly_data.push_back(poly_item->valuedouble);
                        poly_item = poly_item->next;
                    }
                    double *xy = poly_data.data();
                    RLE R;
                    rleFrPoly(&R, xy, int(poly_data.size()) / 2, hw.first, hw.second);
                    rles.push_back(R);
                    polys = polys->next;
                }
                rleMerge(rles.data(), MR, rles.size(), 0);
                for (int i = 0; i < rles.size(); i++) {
                    rleFree(&rles[i]);
                } 
            }

            uint rle_area;
            rleArea(MR, 1, &rle_area);

            _gts[imgId][catId].push_back(
                GtInstance(id->valueint, imgId, catId,
                x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                double(rle_area), isCrowd, isCrowd,
                MR)
            );
        }
        else {
            _gts[imgId][catId].push_back(
                GtInstance(id->valueint, imgId, catId,
                x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                area->valuedouble, isCrowd, isCrowd)
            );

        }
        if (isCrowd == 0) {
            if (cat2cnt.find(catId) == cat2cnt.end()) {
                cat2cnt[catId] = 0;
            }
            cat2cnt[catId] ++;
        }
        ann_item = ann_item->next;
    }

    cJSON* gt_cats = cJSON_GetObjectItem(gt_root, "categories");
    cJSON* cat_item = gt_cats->child;
    while (cat_item != NULL) {
        cJSON* id = cJSON_GetObjectItem(cat_item, "id");
        cJSON* name = cJSON_GetObjectItem(cat_item, "name");
        cat_item = cat_item->next;
        if (params.agnostic) {
            params.catIds.push_back(1);
            params.catNames.push_back("object");
            catname2id[name->valuestring] = 1;
        }
        else {
            params.catIds.push_back(id->valueint);
            params.catNames.push_back(name->valuestring);
            catname2id[name->valuestring] = id->valueint;
        }
    }
}

void COCOEval::load_dts(string dt_path) {
    _dt_path = dt_path;
    ifstream dt_file(dt_path);
    string dt_json((istreambuf_iterator<char>(dt_file)),
                   istreambuf_iterator<char>());

    cJSON* dts = cJSON_Parse(dt_json.c_str());
    cJSON* item = dts->child;

    int i = 0;
    while (item != NULL) {
        cJSON* image_id = cJSON_GetObjectItem(item, "image_id");
        cJSON* category_id = cJSON_GetObjectItem(item, "category_id");
        cJSON* bbox = cJSON_GetObjectItem(item, "bbox");
        cJSON* score = cJSON_GetObjectItem(item, "score");
        cJSON* x = bbox->child;
        cJSON* y = x->next;
        cJSON* w = y->next;
        cJSON* h = w->next;

        int imgId = image_id->valueint;
        int catId = category_id->valueint;
        if (params.agnostic)
            catId = 1;

        if (_iou_type) {
            cJSON* segmentation = cJSON_GetObjectItem(item, "segmentation");
            cJSON* counts = cJSON_GetObjectItem(segmentation, "counts");
            cJSON* size = cJSON_GetObjectItem(segmentation, "size");
            int height = size->child->valueint;
            int width = size->child->next->valueint;
            string rle = counts->valuestring; 
            RLE *R = new RLE();
            rleFrString(R, rle.c_str(), height, width);
            _dts[imgId][catId].push_back(
                DtInstance(i+1, imgId, catId,
                x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                score->valuedouble, false,
                R)
            );
        }
        else {
            _dts[imgId][catId].push_back(
                DtInstance(i+1, imgId, catId,
                x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                score->valuedouble, false)
            );
        }
        item = item->next;
        i++;
    }
}

void COCOEval::load_dts_tsv(string dt_path) {
    _dt_path = dt_path;
    ifstream dt_file(dt_path);
    string s;
    int i = 0;

    int ccc = 0;
    while (getline(dt_file, s)) {
        stringstream ss(s);
        string image_id_str;
        string json_data;
        getline(ss, image_id_str, '\t');
        getline(ss, json_data);

        int imgId = stoi(image_id_str); 
        cJSON* dts = cJSON_Parse(json_data.c_str());
        cJSON* item = dts->child;

        ccc++;
        while (item != NULL) {
            cJSON* category = cJSON_GetObjectItem(item, "class");
            cJSON* score = cJSON_GetObjectItem(item, "conf");
            cJSON* bbox = cJSON_GetObjectItem(item, "rect");

            cJSON* x = bbox->child;
            cJSON* y = x->next;
            cJSON* w = y->next;
            cJSON* h = w->next;

            if (score->valuedouble >= det_thre) {
                int catId = catname2id[category->valuestring];
                if (_iou_type) {
                    cJSON* segmentation = cJSON_GetObjectItem(item, "mask_rle");
                    cJSON* counts = cJSON_GetObjectItem(segmentation, "counts");
                    cJSON* size = cJSON_GetObjectItem(segmentation, "size");
                    int height = size->child->valueint;
                    int width = size->child->next->valueint;
                    string rle = counts->valuestring; 
                    RLE *R = new RLE();
                    rleFrString(R, rle.c_str(), height, width);
                    _dts[imgId][catId].push_back(
                        DtInstance(i+1, imgId, catId,
                        x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                        score->valuedouble, false,
                        R)
                    );
                }
                else {
                    _dts[imgId][catId].push_back(
                        DtInstance(i+1, imgId, catId,
                        x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                        score->valuedouble, false)
                    );
                }
                i++;
            }
            item = item->next;
        }
    }
}

void COCOEval::evaluate() {

    int imgNum = params.imgIds.size();
    int catNum = params.catIds.size();
    int areaNum = params.areaRng.size();

    int maxDet = params.maxDets[params.maxDets.size()-1];

    int T = params.iouThrs.size();
    int R = params.recThrs.size();
    int K = params.catIds.size();
    int A = params.areaRng.size();
    int M = params.maxDets.size();

    precision = new double[T*R*K*A*M];
    recall = new double[T*K*A*M];

    for (int i = 0; i < T*R*K*A*M; i++) {
        precision[i] = -1;
    }
    for (int i = 0; i < T*K*A*M; i++) {
        recall[i] = -1;
    }

    pthread_t tids[params.poolsize];
    threadp p[params.poolsize];
    for (int i = 0; i < params.poolsize; i++) {
        p[i] = threadp(this, i);
        int ret = pthread_create(&tids[i], NULL, evaluateImgStatic, (void*)&p[i]);
    }
    for (int i = 0; i < params.poolsize; i++) {
        pthread_join(tids[i], NULL);
    }
}

void* COCOEval::evaluateImgStatic(void *args) {
    COCOEval* obj = ((threadp*)args)->p;
    int pid = ((threadp*)args)->k;
    int K = obj->params.catIds.size();
    for (int i = 0; i < K; i++) {
        if (i % obj->params.poolsize == pid) {
            obj->evaluateImg(i);
        }
    }
    return NULL;
}

void COCOEval::evaluateImg(int k) {
    int catId = params.catIds[k];

    int T = params.iouThrs.size();
    int R = params.recThrs.size();
    int K = params.catIds.size();
    int A = params.areaRng.size();
    int M = params.maxDets.size();
    int imgNum = params.imgIds.size();
    int maxDet = params.maxDets[params.maxDets.size()-1];

    double **iou = new double*[imgNum];
    for (int i = 0; i < imgNum; i++)
        iou[i] = NULL;

    for (int a = 0; a < A; a++) {

        vector<EvalResult> E;
        vector<double> aRng = params.areaRng[a];

        for (int i = 0; i < imgNum; i++) {

            int imgId = params.imgIds[i];
            
            vector<GtInstance> gt_ori, gt;
            vector<DtInstance> dt_ori, dt;
            if (
                (_gts.find(imgId) != _gts.end()) &&
                (_gts[imgId].find(catId) != _gts[imgId].end())
            ) {
                gt_ori = _gts[imgId][catId];
            }

            if (
                (_dts.find(imgId) != _dts.end()) &&
                (_dts[imgId].find(catId) != _dts[imgId].end())
            ) {
                dt_ori = _dts[imgId][catId];
            }

            int gtNum = gt_ori.size();
            int dtNum = dt_ori.size();

            if (gtNum == 0 && dtNum == 0) {
                continue;
            }

            vector<pair<int, int> > gig;
            for (int i = 0; i < gtNum; i++) {
                if (
                    gt_ori[i].ignore ||
                    gt_ori[i].area < aRng[0] ||
                    gt_ori[i].area > aRng[1]
                )
                    gig.push_back(make_pair(1, i));
                else
                    gig.push_back(make_pair(0, i));
            }
            merge_sort_gig(gig, 0, gig.size()-1);
            for (int i = 0; i < gtNum; i++) {
                gt.push_back(gt_ori[gig[i].second]);
            }

            vector<pair<double, int> > dscore;
            for (int i = 0; i < dtNum; i++) {
                dscore.push_back(make_pair(
                    dt_ori[i].score, i
                )); 
            }
            merge_sort_dscore(dscore, 0, dscore.size()-1);
            if (dtNum > maxDet) dtNum = maxDet;
            for (int i = 0; i < dtNum; i++) {
                dt.push_back(dt_ori[dscore[i].second]);
            }

            vector<int> iscrowd;
            for (int i = 0; i < gtNum; i++) iscrowd.push_back(gt[i].iscrowd);

            if (iou[i] == NULL)
                iou[i] = computeIoU(imgId, catId);
            vector<vector<double> > ious;
            if (iou[i] != NULL) {
                for (int _i = 0; _i < dtNum; _i++) {
                    vector<double> v;
                    for (int _j = 0; _j < gtNum; _j++) {
                        v.push_back(
                            iou[i][gig[_j].second*dtNum+_i]
                        );
                    }
                    ious.push_back(v);
                }
            }

            int T = params.iouThrs.size();
            int G = gtNum;
            int D = dtNum; 
            vector<vector<int> > gtm;
            vector<vector<int> > dtm;
            vector<int> gtIg;
            for (int i = 0; i < gtNum; i++) {
                gtIg.push_back(gig[i].first);
            }
            vector<vector<int> > dtIg;
            for (int i = 0; i < T; i++) {
                gtm.push_back(vector<int>(G));
                dtm.push_back(vector<int>(D));
                dtIg.push_back(vector<int>(D));
            }

            if (ious.size() != 0) {
                for (int tind = 0; tind < params.iouThrs.size(); tind++) {
                    double t = params.iouThrs[tind];
                    for (int dind = 0; dind < D; dind++) {
                        double iou = t;
                        int m = -1;
                        for (int gind = 0; gind < G; gind++) {
                            if (gtm[tind][gind] > 0 && !iscrowd[gind])
                                continue;
                            if (m > -1 && gtIg[m] == 0 && gtIg[gind] == 1)
                                break;
                            if (ious[dind][gind] < iou)
                                continue;
                            iou = ious[dind][gind];
                            m = gind;
                        }
                        if (m == -1)
                            continue;
                        dtIg[tind][dind] = gtIg[m];
                        dtm[tind][dind] = gt[m].id;
                        gtm[tind][m] = dt[dind].id;
                    }
                }
            }

            for (int i = 0; i < T; i++) {
                for (int j = 0; j < D; j++) {
                    if (
                        dtm[i][j] ==0 && (dt[j].area < aRng[0] || dt[j].area > aRng[1] || dt[j].ignore)
                    )
                    dtIg[i][j] = 1;
                }
            }
            vector<double> score;
            for (int i = 0; i < dtNum; i++) {
                score.push_back(dt[i].score);
            }

            E.push_back(EvalResult(
                imgId, catId, aRng, maxDet,
                dtm, gtm, score, gtIg, dtIg
            ));
        }

        for (int m = 0; m < M; m++) {
            if (E.size() == 0)
                continue;

            int maxDetm = params.maxDets[m];
            vector<pair<double, int> > dtScores;
            int ptr = 0;
            for (int i = 0; i < E.size(); i++) {
                int dsize = E[i].dtScores.size();
                if (dsize > maxDetm) dsize = maxDetm;
                for (int j = 0; j < dsize; j++) {
                    dtScores.push_back(
                        make_pair(E[i].dtScores[j], ptr)
                    );
                    ptr++;
                }                        
            } 

            merge_sort_dscore(dtScores, 0, dtScores.size()-1);

            vector<vector<int> > dtm;
            vector<vector<int> > dtIg;
            vector<int> gtIg;
            for (int i = 0 ; i < E[0].dtm.size(); i++) {
                vector<int> tmp(dtScores.size());
                dtm.push_back(tmp);
                dtIg.push_back(tmp);
            }

            for (int j = 0; j < E[0].dtm.size(); j++) {
                vector<int> tmp;
                for (int i = 0; i < E.size(); i++) {
                    for (int k = 0; k < E[i].dtm[j].size() && k < maxDetm; k++) {
                        tmp.push_back(E[i].dtm[j][k]);
                    }
                }
                for (int i = 0; i < tmp.size(); i++) {
                    dtm[j][i] = tmp[dtScores[i].second];
                }
            }

            for (int j = 0; j < E[0].dtIg.size(); j++) {
                vector<int> tmp;
                for (int i = 0; i < E.size(); i++) {
                    for (int k = 0; k < E[i].dtIg[j].size() && k < maxDetm; k++) {
                        tmp.push_back(E[i].dtIg[j][k]);
                    }
                }
                for (int i = 0; i < tmp.size(); i++) {
                    dtIg[j][i] = tmp[dtScores[i].second];
                }
            }

            for (int i = 0; i < E.size(); i++) {
                for (int j = 0; j < E[i].gtIg.size(); j++) {
                    gtIg.push_back(E[i].gtIg[j]);
                }
            }

            int npig = 0;
            for (int i = 0; i < gtIg.size(); i++) {
                if (gtIg[i] == 0) npig++;
            }
            if (npig == 0) continue;

            vector<vector<double> > tps;
            vector<vector<double> > fps;

            for (int i = 0; i < dtIg.size(); i++) {
                vector<double> tp;
                vector<double> fp;
                for (int j = 0; j < dtIg[i].size(); j++) {
                    dtIg[i][j] = 1 - dtIg[i][j];
                    if (dtm[i][j] && dtIg[i][j]) tp.push_back(1);
                    else tp.push_back(0);
                    if (!dtm[i][j] && dtIg[i][j]) fp.push_back(1);
                    else fp.push_back(0);
                }
                tps.push_back(tp);
                fps.push_back(fp);
            }

            for (int i = 0; i < tps.size(); i++) {
                for (int j = 1; j < tps[i].size(); j++) {
                    tps[i][j] += tps[i][j-1];
                    fps[i][j] += fps[i][j-1];
                }
            }

            double eps = 2.220446049250313e-16;
            for (int t = 0; t < tps.size(); t++) {
                vector<double> &tp = tps[t];
                vector<double> &fp = fps[t];
                int nd = tp.size();
                double *rc = new double[nd];
                double *pr = new double[nd]; 
                for (int i = 0; i < nd; i++) {
                    rc[i] = tp[i] / npig;
                    pr[i] = tp[i] / (fp[i] + tp[i] + eps);
                }  
                double q[R]; memset(q, 0, sizeof(q));

                int offset = t*K*A*M + k*A*M + a*M + m;
                if (nd)
                    recall[offset] = rc[nd-1];
                else
                    recall[offset] = 0;

                for (int i = nd-1; i > 0; i--) {
                    if (pr[i] > pr[i-1])
                        pr[i-1] = pr[i];
                }
                int inds[R];
                int ptr1 = 0, ptr2 = 0;
                while (ptr1 < nd && ptr2 < R) {
                    if (rc[ptr1] <= params.recThrs[ptr2]) {
                        ptr1++;
                    }
                    else {
                        inds[ptr2] = ptr1;
                        ptr2++;
                    }
                }
                while (ptr2 < R) {
                    inds[ptr2++] = nd;
                }

                for (int i = 0; i < R; i++) {
                    if (inds[i] < dtScores.size()) {
                        q[i] = pr[inds[i]];
                    }
                    else
                        break;
                }
                for (int i = 0; i < R; i++) {
                    offset = t*R*K*A*M + i*K*A*M + k*A*M + a*M + m;
                    precision[offset] = q[i];
                }
                delete rc;
                delete pr;
            }

        }
    }

    for (int i = 0; i < imgNum; i++)
        if (iou[i] != NULL)
            delete iou[i];
    delete iou;
}


double* COCOEval::computeIoU(int imgId, int catId) {
    if (
        (_gts.find(imgId) == _gts.end()) ||
        (_gts[imgId].find(catId) == _gts[imgId].end())
    ) {
        return NULL;
    }

    if (
        (_dts.find(imgId) == _dts.end()) ||
        (_dts[imgId].find(catId) == _dts[imgId].end())
    ) {
        return NULL;
    }

    vector<GtInstance> &gt = _gts[imgId][catId];
    vector<DtInstance> &dt = _dts[imgId][catId]; 

    merge_sort_dt(dt, 0, dt.size()-1);
    int dtnum = dt.size();
    if (dtnum > params.maxDets[params.maxDets.size()-1]) {
        dtnum = params.maxDets[params.maxDets.size()-1];
    }
    int gtnum = gt.size();
    bytee *iscrowd = new bytee[gtnum];
    double *iou = new double[dtnum*gtnum];
    
    if (_iou_type) {
        RLE *dtrle = new RLE[dtnum];
        RLE *gtrle = new RLE[gtnum];
        for (int i = 0; i < dtnum; i++) {
            dtrle[i] = *(dt[i].segmentation);
        }
        for (int i = 0; i < gtnum; i++) {
            gtrle[i] = *(gt[i].segmentation);
            iscrowd[i] = gt[i].iscrowd;
        }
        rleIou(dtrle, gtrle, (unsigned long)dtnum, (unsigned long)gtnum, iscrowd, iou);
        delete dtrle;
        delete gtrle;
    }
    else {
        double *dtbb = new double[dtnum*4];
        double *gtbb = new double[gtnum*4];
        for (int i = 0; i < dtnum; i++) {
            dtbb[i*4] = dt[i].bbox[0];
            dtbb[i*4+1] = dt[i].bbox[1];
            dtbb[i*4+2] = dt[i].bbox[2];
            dtbb[i*4+3] = dt[i].bbox[3];
        }
        for (int i = 0; i < gtnum; i++) {
            gtbb[i*4] = gt[i].bbox[0];
            gtbb[i*4+1] = gt[i].bbox[1];
            gtbb[i*4+2] = gt[i].bbox[2];
            gtbb[i*4+3] = gt[i].bbox[3];
            iscrowd[i] = gt[i].iscrowd;
        }
        bbIou(dtbb, gtbb, (unsigned long)dtnum, (unsigned long)gtnum, iscrowd, iou);
        delete dtbb;
        delete gtbb;
    }
    delete iscrowd;
    return iou;
}


map<string, vector<double> > COCOEval::accumulate() {
    int T = params.iouThrs.size();
    int R = params.recThrs.size();
    int K = params.catIds.size();
    int A = params.areaRng.size();
    int M = params.maxDets.size();

    clock_t start, end;
    double dur;
    start = clock();

    int summarize_ap[12] = {1,1,1,1,1,1,0,0,0,0,0,0};
    double summarize_iouThr[12] = {-1,0.5,0.75,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    string summarize_areaRng[12] = {
        "all", "all", "all", "small", "medium", "large", "all", "all", "all", "small", "medium", "large"
    };
    int summarize_maxDets[12] = {100,100,100,100,100,100,1,10,100,100,100,100};

    for (int i = 0; i < 12; i++) {
        int ap = summarize_ap[i];
        double iouThr = summarize_iouThr[i];
        string areaRng = summarize_areaRng[i];
        int maxDets = summarize_maxDets[i]; 

        char titleStr[1000], typeStr[1000], iouStr[1000];

        int aind;
        for (int j = 0; j < params.areaRngLbl.size(); j++) {
            if (params.areaRngLbl[j] == areaRng)
                aind = j;
        }

        int mind;
        for (int j = 0; j < params.maxDets.size(); j++) {
            if (params.maxDets[j] == maxDets)
                mind = j;
        }

        double sum_p = 0;
        double sum = 0;
        double mean_s;
        int offset;
        if (ap == 1) {
            if (iouThr != -1) {
                int tind;
                for (int j = 0; j < params.iouThrs.size(); j++) {
                    if (params.iouThrs[j] == iouThr)
                        tind = j;
                }
                for (int r = 0; r < R; r++) {
                    for (int k = 0; k < K; k++) {
                        offset = tind*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                        if (precision[offset] != -1) {
                            sum_p += precision[offset];
                            sum += 1;
                        }
                    }
                }
            } 
            else {
                for (int t = 0; t < T; t++) {
                    for (int r = 0; r < R; r++) {
                        for (int k = 0; k < K; k++) {
                            offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                            if (precision[offset] != -1) {
                                sum_p += precision[offset];
                                sum += 1;
                            }
                        }
                    }
                }
            }
            if (sum != 0) mean_s = sum_p / sum;
            else mean_s = -1;
        }
        else {
            if (iouThr != -1) {
                int tind;
                for (int j = 0; j < params.iouThrs.size(); j++) {
                    if (params.iouThrs[j] == iouThr)
                        tind = j;
                }
                for (int k = 0; k < K; k++) {
                    offset = tind*K*A*M + k*A*M + aind*M + mind;
                    if (recall[offset] != -1) {
                        sum_p += recall[offset];
                        sum += 1;
                    }
                }
            } 
            else {
                for (int t = 0; t < T; t++) {
                    for (int k = 0; k < K; k++) {
                        offset = t*K*A*M + k*A*M + aind*M + mind;
                        if (recall[offset] != -1) {
                            sum_p += recall[offset];
                            sum += 1;
                        }
                    }
                }
            }
            if (sum != 0) mean_s = sum_p / sum;
            else mean_s = -1;
        }
    }

    int aind = 0, mind = 2, offset;
    double ap, ap_sum;
    double ap50, ap50_sum;
    double ratp70, ratp70_sum;
    double ratp50, ratp50_sum;
    double maxprecision;
    double precision_value, recall_value;
    double precision50_value, recall50_value;
    double precision95_value, recall95_value;

    map<string, vector<double> > result;
    vector<double> vector_ap;
    vector<double> vector_ap50;
    vector<double> vector_ap_small;
    vector<double> vector_ap_middle;
    vector<double> vector_ap_large;
    vector<double> vector_ap50_small;
    vector<double> vector_ap50_middle;
    vector<double> vector_ap50_large;
    vector<double> vector_ratp70;
    vector<double> vector_ratp50;
    vector<double> vector_maxprecision;
    vector<double> vector_precision;
    vector<double> vector_recall;
    vector<double> vector_precision50;
    vector<double> vector_recall50;
    vector<double> vector_precision95;
    vector<double> vector_recall95;

    int K_loop = K;
    if (params.agnostic) K_loop = 1;

    for (int k = 0; k < K_loop; k++) {
        ap = 0; ap_sum = 0;
        ap50 = 0; ap50_sum = 0;
        ratp70 = -1; ratp50 = -1; maxprecision = -1;
        for (int r = 0; r < R; r++) {
            //AP
            for (int t = 0; t < T; t++) {
                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                if (precision[offset] != -1) {
                    ap += precision[offset];
                    ap_sum += 1;
                } 
            }        
            //AP50
            int t = 0;
            offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
            if (precision[offset] != -1) {
                ap50 += precision[offset];
                ap50_sum += 1;
            }
            //R@P70 R@P50 MaxPrecision
            if (precision[offset] > -0.5) {
                if (precision[offset] > 0.7) {
                    ratp70 = params.recThrs[r]; 
                }
                if (precision[offset] > 0.5) {
                    ratp50 = params.recThrs[r];
                }
                if (precision[offset] > maxprecision) {
                    maxprecision = precision[offset];
                }
            }
        }
        ap = ap / ap_sum;
        ap50 = ap50 / ap50_sum;

        for (int r = R-1; r >= 0; r--) {
            offset = 0*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
            if (precision[offset] > 0) {
                precision50_value = precision[offset];
                break;
            }
        }
        offset = 0*K*A*M + k*A*M + aind*M + mind;
        recall50_value = recall[offset];

        for (int r = R-1; r >= 0; r--) {
            offset = (T-1)*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
            if (precision[offset] > 0) {
                precision95_value = precision[offset];
                break;
            }
        }
        offset = (T-1)*K*A*M + k*A*M + aind*M + mind;
        recall95_value = recall[offset];

        precision_value = 0;
        recall_value = 0;
        double precision_value_tmp, recall_value_tmp;
        for (int t = 0; t < T; t++) {
            for (int r = R-1; r >= 0; r--) {
                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                if (precision[offset] > 0) {
                    precision_value_tmp = precision[offset];
                    break;
                }
            }
            precision_value += precision_value_tmp;
            offset = t*K*A*M + k*A*M + aind*M + mind;
            recall_value_tmp = recall[offset];
            recall_value += recall_value_tmp; 
        }
        precision_value /= T;
        recall_value /= T;

        vector_ap.push_back(ap);
        vector_ap50.push_back(ap50);
        vector_ratp70.push_back(ratp70);
        vector_ratp50.push_back(ratp50);
        vector_maxprecision.push_back(maxprecision);
        vector_precision.push_back(precision_value);
        vector_recall.push_back(recall_value);
        vector_precision50.push_back(precision50_value);
        vector_recall50.push_back(recall50_value);
        vector_precision95.push_back(precision95_value);
        vector_recall95.push_back(recall95_value);
    }

    vector<double> *v_ap;
    vector<double> *v_ap50;
    for (aind = 1; aind <= 3; aind++) {
        if (aind == 1) {
            v_ap = &vector_ap_small;
            v_ap50 = &vector_ap50_small;
        }
        else if (aind == 2) {
            v_ap = &vector_ap_middle;
            v_ap50 = &vector_ap50_middle;
        }
        else if (aind == 3) {
            v_ap = &vector_ap_large;
            v_ap50 = &vector_ap50_large;
        }
        for (int k = 0; k < K_loop; k++) {
            ap = 0; ap_sum = 0;
            ap50 = 0; ap50_sum = 0;
            for (int r = 0; r < R; r++) {
                //AP
                for (int t = 0; t < T; t++) {
                    offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                    if (precision[offset] != -1) {
                        ap += precision[offset];
                        ap_sum += 1;
                    } 
                }        
                //AP50
                int t = 0;
                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                if (precision[offset] != -1) {
                    ap50 += precision[offset];
                    ap50_sum += 1;
                }
            }
            ap = ap / ap_sum;
            ap50 = ap50 / ap50_sum;

            v_ap->push_back(ap);
            v_ap50->push_back(ap50);
        }
    }

    result["ap"] = vector_ap;
    result["ap50"] = vector_ap50;
    result["ratp70"] = vector_ratp70;
    result["ratp50"] = vector_ratp50;
    result["maxprecision"] = vector_maxprecision;
    result["precision"] = vector_precision;
    result["recall"] = vector_recall;
    result["ap_small"] = vector_ap_small;
    result["ap_middle"] = vector_ap_middle;
    result["ap_large"] = vector_ap_large;
    result["ap50_small"] = vector_ap50_small;
    result["ap50_middle"] = vector_ap50_middle;
    result["ap50_large"] = vector_ap50_large;
    result["precision50"] = vector_precision50;
    result["recall50"] = vector_recall50;
    result["precision95"] = vector_precision95;
    result["recall95"] = vector_recall95;


    delete precision;
    delete recall;
    return result;
}

LVISEval::LVISEval(string gt_path, string dt_path, string type, float thre, string iou_type) {
    _iou_type = iou_type == "segm";
    vector<int> maxDets = {1000000000};
    params.maxDets = maxDets;
    det_thre = thre;

    clock_t start, end;
    double dur;
    start = clock();
    load_gts(gt_path);
    end = clock();
    dur = (double)(end - start)/CLOCKS_PER_SEC;
    printf("load gts cost %lf\n", dur);


    start = clock();

    if (type == "json") {
        load_dts(dt_path);
    }
    else if (type == "tsv") {
        load_dts_tsv(dt_path);
    }
    end = clock();
    dur = (double)(end - start)/CLOCKS_PER_SEC;
    printf("load dts cost %lf\n", dur);
}


void LVISEval::load_gts(string gt_path) {
    ifstream gt_file(gt_path);
    string gt_json((istreambuf_iterator<char>(gt_file)),
                   istreambuf_iterator<char>());

    cJSON* gt_root = cJSON_Parse(gt_json.c_str());
    cJSON* gt_images = cJSON_GetObjectItem(gt_root, "images");
    cJSON* img_item = gt_images->child;
    map<int, pair<int, int> > imgid2hw;
    while (img_item != NULL) {
        cJSON* id = cJSON_GetObjectItem(img_item, "id");
        cJSON* width = cJSON_GetObjectItem(img_item, "width");
        cJSON* height = cJSON_GetObjectItem(img_item, "height");
        imgid2hw[id->valueint] = make_pair(height->valueint, width->valueint);
        params.imgIds.push_back(id->valueint);

        cJSON* neg = cJSON_GetObjectItem(img_item, "neg_category_ids");
        cJSON* neg_item = neg->child;
        while (neg_item != NULL) {
            img_nl[id->valueint].insert(neg_item->valueint);
            neg_item = neg_item->next;
        }    

        cJSON* nel = cJSON_GetObjectItem(img_item, "not_exhaustive_category_ids");
        cJSON* nel_item = nel->child;
        set<int> &nel_s = img_nel[id->valueint];
        while (nel_item != NULL) {
            nel_s.insert(nel_item->valueint);
            nel_item = nel_item->next;
        }    
        
        img_item = img_item->next;
    }


    cJSON* gt_annotations = cJSON_GetObjectItem(gt_root, "annotations");
    cJSON* ann_item = gt_annotations->child;
    while (ann_item != NULL) {
        cJSON* id = cJSON_GetObjectItem(ann_item, "id");
        cJSON* image_id = cJSON_GetObjectItem(ann_item, "image_id");
        cJSON* category_id = cJSON_GetObjectItem(ann_item, "category_id");
        cJSON* bbox = cJSON_GetObjectItem(ann_item, "bbox");
        cJSON* area = cJSON_GetObjectItem(ann_item, "area");
        cJSON* x = bbox->child;
        cJSON* y = x->next;
        cJSON* w = y->next;
        cJSON* h = w->next;

        if (_iou_type) {
            cJSON* segmentation = cJSON_GetObjectItem(ann_item, "segmentation");
            pair<int, int> hw = imgid2hw[image_id->valueint];

            RLE *MR = new RLE();
            cJSON* counts = cJSON_GetObjectItem(segmentation, "counts");
            if (counts != NULL) {
                vector<uint> cnts;
                cJSON *counts_item = counts->child;
                while (counts_item != NULL) {
                    cnts.push_back(counts_item->valueint);
                    counts_item = counts_item->next;
                }
                rleInit(MR, hw.first, hw.second, cnts.size(), cnts.data());
            }
            else {
                cJSON* polys = segmentation->child;
                vector<RLE> rles;
                while (polys != NULL) {
                    cJSON* poly_item = polys->child;
                    vector<double> poly_data;
                    while (poly_item != NULL) {
                        poly_data.push_back(poly_item->valuedouble);
                        poly_item = poly_item->next;
                    }
                    double *xy = poly_data.data();
                    RLE R;
                    rleFrPoly(&R, xy, int(poly_data.size()) / 2, hw.first, hw.second);
                    rles.push_back(R);
                    polys = polys->next;
                }
                rleMerge(rles.data(), MR, rles.size(), 0);
                for (int i = 0; i < rles.size(); i++) {
                    rleFree(&rles[i]);
                } 
            }

            uint rle_area;
            rleArea(MR, 1, &rle_area);

            _gts[image_id->valueint][category_id->valueint].push_back(
                GtInstance(id->valueint, image_id->valueint, category_id->valueint,
                x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                double(rle_area), 0, 0, MR)
            );
        }

        else {
            _gts[image_id->valueint][category_id->valueint].push_back(
                GtInstance(id->valueint, image_id->valueint, category_id->valueint,
                x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                area->valuedouble, 0, 0)
            );
        }
        img_pl[image_id->valueint].insert(category_id->valueint);
        ann_item = ann_item->next;
    }


    cJSON* gt_cats = cJSON_GetObjectItem(gt_root, "categories");
    cJSON* cat_item = gt_cats->child;
    int cid = 0;
    while (cat_item != NULL) {
        cJSON* id = cJSON_GetObjectItem(cat_item, "id");
        cJSON* name = cJSON_GetObjectItem(cat_item, "name");
        cJSON* freq = cJSON_GetObjectItem(cat_item, "frequency");
        params.catIds.push_back(id->valueint);
        cat_item = cat_item->next;
        catname2id[name->valuestring] = id->valueint;
        freq_groups[freq->valuestring].push_back(cid++);
    }
}

void LVISEval::load_dts(string dt_path) {
    _dt_path = dt_path;
    ifstream dt_file(dt_path);
    string dt_json((istreambuf_iterator<char>(dt_file)),
                   istreambuf_iterator<char>());

    cJSON* dts = cJSON_Parse(dt_json.c_str());
    cJSON* item = dts->child;

    int i = 0;
    while (item != NULL) {
        cJSON* image_id = cJSON_GetObjectItem(item, "image_id");
        cJSON* category_id = cJSON_GetObjectItem(item, "category_id");

        int imgId = image_id->valueint;
        int catId = category_id->valueint;
        set<int> &pl = img_pl[imgId];
        set<int> &nl = img_nl[imgId];
        if (pl.find(catId) == pl.end() &&
            nl.find(catId) == nl.end()) {
            item = item->next;
            continue;
        }

        cJSON* bbox = cJSON_GetObjectItem(item, "bbox");
        cJSON* score = cJSON_GetObjectItem(item, "score");
        cJSON* x = bbox->child;
        cJSON* y = x->next;
        cJSON* w = y->next;
        cJSON* h = w->next;

        bool ignore = false;
        set<int> &nel = img_nel[imgId];
        if (nel.find(catId) != nel.end()) ignore = true;

        if (score->valuedouble > det_thre) {
            _dts[imgId][catId].push_back(
                DtInstance(i+1, imgId, catId,
                x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                score->valuedouble, ignore)
            );
        }
        item = item->next;
        i++;
    }
}

void LVISEval::load_dts_tsv(string dt_path) {
    _dt_path = dt_path;
    ifstream dt_file(dt_path);
    string s;
    int i = 0;
    while (getline(dt_file, s)) {
        stringstream ss(s);
        string image_id_str;
        string json_data;
        getline(ss, image_id_str, '\t');
        getline(ss, json_data);

        int image_id = stoi(image_id_str); 
        cJSON* dts = cJSON_Parse(json_data.c_str());
        cJSON* item = dts->child;

        set<int> &pl = img_pl[image_id];
        set<int> &nl = img_nl[image_id];

        while (item != NULL) {
            cJSON* category = cJSON_GetObjectItem(item, "class");
            int category_id = catname2id[category->valuestring];
            if (pl.find(category_id) == pl.end() &&
                nl.find(category_id) == nl.end()) {
                item = item->next;
                continue;
            }

            cJSON* score = cJSON_GetObjectItem(item, "conf");
            cJSON* bbox = cJSON_GetObjectItem(item, "rect");
            cJSON* x = bbox->child;
            cJSON* y = x->next;
            cJSON* w = y->next;
            cJSON* h = w->next;

            bool ignore = false;
            set<int> &nel = img_nel[image_id];
            if (nel.find(category_id) != nel.end()) ignore = true;

            if (score->valuedouble > det_thre) {
                if (_iou_type) {
                    cJSON* segmentation = cJSON_GetObjectItem(item, "mask_rle");
                    cJSON* counts = cJSON_GetObjectItem(segmentation, "counts");
                    cJSON* size = cJSON_GetObjectItem(segmentation, "size");
                    int height = size->child->valueint;
                    int width = size->child->next->valueint;
                    string rle = counts->valuestring; 
                    RLE *R = new RLE();
                    rleFrString(R, rle.c_str(), height, width);
                    _dts[image_id][category_id].push_back(
                        DtInstance(i+1, image_id, category_id,
                        x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                        score->valuedouble, ignore, R)
                    );
                }
                else {
                    _dts[image_id][category_id].push_back(
                        DtInstance(i+1, image_id, category_id,
                        x->valuedouble, y->valuedouble, w->valuedouble, h->valuedouble,
                        score->valuedouble, ignore)
                    );
                }
            }

            item = item->next;
            i++;
        }
    }
}

map<string, vector<double> > LVISEval::accumulate() {
    int T = params.iouThrs.size();
    int R = params.recThrs.size();
    int K = params.catIds.size();
    int A = params.areaRng.size();
    int M = params.maxDets.size();

    clock_t start, end;
    double dur;
    start = clock();

    int summarize_ap[13] = {1,1,1,1,1,1,1,1,1,0,0,0,0};
    double summarize_iouThr[13] = {-1,0.5,0.75,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    string summarize_areaRng[13] = {
        "all", "all", "all", "small", "medium", "large", "all", "all", "all", "all", "small", "medium", "large"
    };
    string summarize_imgRng[13] = {
        "all", "all", "all", "all", "all", "all", "r", "c", "f", "all", "all", "all", "all"
    };

    for (int i = 0; i < 13; i++) {
        int ap = summarize_ap[i];
        double iouThr = summarize_iouThr[i];
        string areaRng = summarize_areaRng[i];
        string imgRng = summarize_imgRng[i];
        int maxDets = -1; 

        char titleStr[1000], typeStr[1000], iouStr[1000];

        int aind;
        for (int j = 0; j < params.areaRngLbl.size(); j++) {
            if (params.areaRngLbl[j] == areaRng)
                aind = j;
        }

        int mind = params.maxDets.size() - 1;

        double sum_p = 0;
        double sum = 0;
        double mean_s;
        int offset;
        if (ap == 1) {
            if (iouThr != -1) {
                int tind;
                for (int j = 0; j < params.iouThrs.size(); j++) {
                    if (params.iouThrs[j] == iouThr)
                        tind = j;
                }
                for (int r = 0; r < R; r++) {
                    if (imgRng == "all") {
                        for (int k = 0; k < K; k++) {
                            offset = tind*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                            if (precision[offset] != -1) {
                                sum_p += precision[offset];
                                sum += 1;
                            }
                        }
                    }
                    else {
                        for (int _k = 0; _k < freq_groups[summarize_imgRng[i]].size(); _k++) {
                            int k = freq_groups[summarize_imgRng[i]][_k];
                            offset = tind*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                            if (precision[offset] != -1) {
                                sum_p += precision[offset];
                                sum += 1;
                            }
                        }
                    }
                }
            } 
            else {
                for (int t = 0; t < T; t++) {
                    for (int r = 0; r < R; r++) {
                        if (imgRng == "all") {
                            for (int k = 0; k < K; k++) {
                                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                                if (precision[offset] != -1) {
                                    sum_p += precision[offset];
                                    sum += 1;
                                }
                            }
                        }
                        else {
                            for (int _k = 0; _k < freq_groups[summarize_imgRng[i]].size(); _k++) {
                                int k = freq_groups[summarize_imgRng[i]][_k];
                                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                                if (precision[offset] != -1) {
                                    sum_p += precision[offset];
                                    sum += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            if (iouThr != -1) {
                int tind;
                for (int j = 0; j < params.iouThrs.size(); j++) {
                    if (params.iouThrs[j] == iouThr)
                        tind = j;
                }
                for (int k = 0; k < K; k++) {
                    offset = tind*K*A*M + k*A*M + aind*M + mind;
                    if (recall[offset] != -1) {
                        sum_p += recall[offset];
                        sum += 1;
                    }
                }
            } 
            else {
                for (int t = 0; t < T; t++) {
                    for (int k = 0; k < K; k++) {
                        offset = t*K*A*M + k*A*M + aind*M + mind;
                        if (recall[offset] != -1) {
                            sum_p += recall[offset];
                            sum += 1;
                        }
                    }
                }
            }
            if (sum != 0) mean_s = sum_p / sum;
            else mean_s = -1;
        } 
        if (sum != 0) mean_s = sum_p / sum;
        else mean_s = -1;
        printf("%-18s %s @[ IoU=%-9s | area=%6s | maxDets=%3d catIds=%3s] = %0.3f\n", titleStr, typeStr, iouStr, areaRng.c_str(), maxDets, imgRng.c_str(), mean_s);

    }

    int aind = 0, mind = 0, offset;
    double ap, ap_sum;
    double ap50, ap50_sum;
    double ratp70, ratp70_sum;
    double ratp50, ratp50_sum;
    double maxprecision;
    double precision_value, recall_value;
    double precision50_value, recall50_value;
    double precision95_value, recall95_value;

    map<string, vector<double> > result;
    vector<double> vector_ap;
    vector<double> vector_ap_small;
    vector<double> vector_ap_middle;
    vector<double> vector_ap_large;
    vector<double> vector_ap_r;
    vector<double> vector_ap_c;
    vector<double> vector_ap_f;
    vector<double> vector_ap50;
    vector<double> vector_ap50_small;
    vector<double> vector_ap50_middle;
    vector<double> vector_ap50_large;
    vector<double> vector_ap50_r;
    vector<double> vector_ap50_c;
    vector<double> vector_ap50_f;
    vector<double> vector_ratp70;
    vector<double> vector_ratp50;
    vector<double> vector_maxprecision;
    vector<double> vector_precision;
    vector<double> vector_recall;
    vector<double> vector_precision50;
    vector<double> vector_recall50;
    vector<double> vector_precision95;
    vector<double> vector_recall95;

    int K_loop = K;
    if (params.agnostic) K_loop = 1;

    for (int k = 0; k < K_loop; k++) {
        ap = 0; ap_sum = 0;
        ap50 = 0; ap50_sum = 0;
        ratp70 = -1; ratp50 = -1; maxprecision = -1;
        for (int r = 0; r < R; r++) {
            //AP
            for (int t = 0; t < T; t++) {
                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                if (precision[offset] != -1) {
                    ap += precision[offset];
                    ap_sum += 1;
                } 
            }        
            //AP50
            int t = 0;
            offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
            if (precision[offset] != -1) {
                ap50 += precision[offset];
                ap50_sum += 1;
            }
            //R@P70 R@P50 MaxPrecision
            if (precision[offset] > -0.5) {
                if (precision[offset] > 0.7) {
                    ratp70 = params.recThrs[r]; 
                }
                if (precision[offset] > 0.5) {
                    ratp50 = params.recThrs[r];
                }
                if (precision[offset] > maxprecision) {
                    maxprecision = precision[offset];
                }
            }
        }
        ap = ap / ap_sum;
        ap50 = ap50 / ap50_sum;

        for (int r = R-1; r >= 0; r--) {
            offset = 0*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
            if (precision[offset] > 0) {
                precision50_value = precision[offset];
                break;
            }
        }
        offset = 0*K*A*M + k*A*M + aind*M + mind;
        recall50_value = recall[offset];

        for (int r = R-1; r >= 0; r--) {
            offset = (T-1)*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
            if (precision[offset] > 0) {
                precision95_value = precision[offset];
                break;
            }
        }
        offset = (T-1)*K*A*M + k*A*M + aind*M + mind;
        recall95_value = recall[offset];

        precision_value = 0;
        recall_value = 0;
        double precision_value_tmp, recall_value_tmp;
        for (int t = 0; t < T; t++) {
            for (int r = R-1; r >= 0; r--) {
                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                if (precision[offset] > 0) {
                    precision_value_tmp = precision[offset];
                    break;
                }
            }
            precision_value += precision_value_tmp;
            offset = t*K*A*M + k*A*M + aind*M + mind;
            recall_value_tmp = recall[offset];
            recall_value += recall_value_tmp; 
        }
        precision_value /= T;
        recall_value /= T;


        vector_ap.push_back(ap);
        vector_ap50.push_back(ap50);
        vector_ratp70.push_back(ratp70);
        vector_ratp50.push_back(ratp50);
        vector_maxprecision.push_back(maxprecision);
        vector_precision50.push_back(precision50_value);
        vector_recall50.push_back(recall50_value);
        vector_precision95.push_back(precision95_value);
        vector_recall95.push_back(recall95_value);
        vector_precision.push_back(precision_value);
        vector_recall.push_back(recall_value);
    }

    vector<double> *v_ap;
    vector<double> *v_ap50;
    for (aind = 1; aind <= 3; aind++) {
        if (aind == 1) {
            v_ap = &vector_ap_small;
            v_ap50 = &vector_ap50_small;
        }
        else if (aind == 2) {
            v_ap = &vector_ap_middle;
            v_ap50 = &vector_ap50_middle;
        }
        else if (aind == 3) {
            v_ap = &vector_ap_large;
            v_ap50 = &vector_ap50_large;
        }
        for (int k = 0; k < K_loop; k++) {
            ap = 0; ap_sum = 0;
            ap50 = 0; ap50_sum = 0;
            for (int r = 0; r < R; r++) {
                //AP
                for (int t = 0; t < T; t++) {
                    offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                    if (precision[offset] != -1) {
                        ap += precision[offset];
                        ap_sum += 1;
                    } 
                }        
                //AP50
                int t = 0;
                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                if (precision[offset] != -1) {
                    ap50 += precision[offset];
                    ap50_sum += 1;
                }
            }
            ap = ap / ap_sum;
            ap50 = ap50 / ap50_sum;

            v_ap->push_back(ap);
            v_ap50->push_back(ap50);
        }
    }

    string imgRng[3] = {"r", "c", "f"};
    aind = 0;
    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            v_ap = &vector_ap_r;
            v_ap50 = &vector_ap50_r;
        }
        else if (i == 1) {
            v_ap = &vector_ap_c;
            v_ap50 = &vector_ap50_c;
        }
        else if (i == 2) {
            v_ap = &vector_ap_f;
            v_ap50 = &vector_ap50_f;
        }
        for (int _k = 0; _k < freq_groups[imgRng[i]].size(); _k++) {
            int k = freq_groups[imgRng[i]][_k];
            ap = 0; ap_sum = 0;
            ap50 = 0; ap50_sum = 0;
            for (int r = 0; r < R; r++) {
                //AP
                for (int t = 0; t < T; t++) {
                    offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                    if (precision[offset] != -1) {
                        ap += precision[offset];
                        ap_sum += 1;
                    }
                }
                //AP50
                int t = 0;
                offset = t*R*K*A*M + r*K*A*M + k*A*M + aind*M + mind;
                if (precision[offset] != -1) {
                    ap50 += precision[offset];
                    ap50_sum += 1;
                }
            }
            ap = ap / ap_sum;
            ap50 = ap50 / ap50_sum;

            v_ap->push_back(ap);
            v_ap50->push_back(ap50);
        } 
    }

    result["ap"] = vector_ap;
    result["ap_small"] = vector_ap_small;
    result["ap_middle"] = vector_ap_middle;
    result["ap_large"] = vector_ap_large;
    result["ap_r"] = vector_ap_r;
    result["ap_c"] = vector_ap_c;
    result["ap_f"] = vector_ap_f;
    result["ap50"] = vector_ap50;
    result["ap50_small"] = vector_ap50_small;
    result["ap50_middle"] = vector_ap50_middle;
    result["ap50_large"] = vector_ap50_large;
    result["ap50_r"] = vector_ap50_r;
    result["ap50_c"] = vector_ap50_c;
    result["ap50_f"] = vector_ap50_f;
    result["ratp70"] = vector_ratp70;
    result["ratp50"] = vector_ratp50;
    result["maxprecision"] = vector_maxprecision;
    result["precision"] = vector_precision;
    result["recall"] = vector_recall;
    result["precision50"] = vector_precision50;
    result["recall50"] = vector_recall50;
    result["precision95"] = vector_precision95;
    result["recall95"] = vector_recall95;

    delete precision;
    delete recall;
    return result;
}


