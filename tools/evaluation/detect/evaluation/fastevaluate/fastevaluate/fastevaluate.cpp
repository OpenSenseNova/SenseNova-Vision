#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <Python.h>
#include "cocoeval.h"

using namespace std;

map<string, vector<double> > evaluate(
    const char* _gt_path, 
    const char* _dt_path,
    int agnostic,
    double threshold,
    const char* _eval_type
) {

    string gt_path = _gt_path;
    string dt_path = _dt_path;
    string dt_type;

    int l = dt_path.size();
    if (dt_path[l-4] == 'j' && dt_path[l-3] == 's' && dt_path[l-2] == 'o' && dt_path[l-1] == 'n') {
        dt_type = "json";
    }
    else if (dt_path[l-3] == 't' && dt_path[l-2] == 's' && dt_path[l-1] == 'v') {
        dt_type = "tsv";
    }
    string eval_type = _eval_type;

    if (eval_type == "coco") {
        COCOEval coco = COCOEval(
            gt_path,
            dt_path,
            dt_type,
            agnostic,
            threshold,
            "bbox"
        );
        coco.evaluate();
        return coco.accumulate();
    }
    else if (eval_type == "cocomask") {
        COCOEval coco = COCOEval(
            gt_path,
            dt_path,
            dt_type,
            agnostic,
            threshold,
            "segm"
        );
        coco.evaluate();
        return coco.accumulate();
    }
    else if (eval_type == "lvis") {
        LVISEval lvis = LVISEval(
            gt_path,
            dt_path,
            dt_type,
            threshold,
            "bbox" 
        );
        lvis.evaluate();
        return lvis.accumulate();
    }
    else if (eval_type == "lvismask") {
        LVISEval lvis = LVISEval(
            gt_path,
            dt_path,
            dt_type,
            threshold,
            "segm" 
        );
        lvis.evaluate();
        return lvis.accumulate();
    }
    else {
        printf("Not implementation for eval type {%s}", _eval_type);
        return map<string, vector<double> >();
    }
}

static PyObject* py_evaluate(PyObject* self, PyObject* args) {
    const char* gt_path;
    const char* dt_path;
    int agnostic;
    double threshold;
    const char* eval_type;

    if (!PyArg_ParseTuple(args, "ssids", &gt_path, &dt_path, &agnostic, &threshold, &eval_type)) {
        return NULL;
    }
    map<string, vector<double> > result = evaluate(gt_path, dt_path, agnostic, threshold, eval_type);

    PyObject* dict = PyDict_New();

    map<string, vector<double> >::iterator iter;
    for (iter = result.begin(); iter != result.end(); iter++) {
        vector<double> &v = iter->second;
        PyObject* list = PyList_New(int(v.size()));
        for (int i = 0; i < v.size(); i++) {
            PyList_SetItem(list, i, PyFloat_FromDouble(v[i]));
        }
        PyDict_SetItemString(dict, iter->first.c_str(), list);
    }
    return dict;
}

static PyMethodDef methods[] = {
    {"evaluate", py_evaluate, METH_VARARGS, "evaluate detection metrics"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "fastevaluate",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_fastevaluate(void) {
    return PyModule_Create(&module);
}
