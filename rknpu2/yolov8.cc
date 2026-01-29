// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>

#include "yolov8.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "rknn_api.h"

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_yolov8_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_yolov8_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_yolov8_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
{
    struct timeval start2, end2, start3, end3, start4, end4, start5, end5, start6, end6, start7, end7, start8, end8;
    double time_use2 = 0;
    //printf("inference_yolov8_model rknpu2\n");
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    gettimeofday(&start6, NULL);
    dst_img.size = get_image_size(&dst_img);
    // 结束计时
    gettimeofday(&end6, NULL);

    // 计算耗时（微秒→毫秒）
    double time_use6 = (end6.tv_sec - start6.tv_sec) * 1000.0 +
               (end6.tv_usec - start6.tv_usec) / 1000.0;
    printf("get_image_size 耗时：%.3f ms\n", time_use6);

    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }
    gettimeofday(&start8, NULL);
    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }
    // 结束计时
    gettimeofday(&end8, NULL);

    // 计算耗时（微秒→毫秒）
    double time_use8 = (end8.tv_sec - start8.tv_sec) * 1000.0 +
               (end8.tv_usec - start8.tv_usec) / 1000.0;
    printf("convert_image_with_letterbox 耗时：%.3f ms\n", time_use8);

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = dst_img.virt_addr;
    gettimeofday(&start7, NULL);
    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Set Core Mask
    ret = rknn_set_core_mask(app_ctx->rknn_ctx, RKNN_NPU_CORE_AUTO);
    if (ret < 0)
    {
        printf("rknn_set_core_mask fail! ret=%d\n", ret);
        return -1;
    }else if (ret == 0)
    {
        printf("rknn_set_core_mask success! ret=%d\n", ret);
    }
    // 结束计时
    gettimeofday(&end7, NULL);

    // 计算耗时（微秒→毫秒）
    double time_use7 = (end7.tv_sec - start7.tv_sec) * 1000.0 +
               (end7.tv_usec - start7.tv_usec) / 1000.0;
    printf("rknn_inputs_set 耗时：%.3f ms\n", time_use7);

    // 开始计时
    gettimeofday(&start2, NULL);
    printf("rknn_run\n");


    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    // 结束计时
    gettimeofday(&end2, NULL);

    // 计算耗时（微秒→毫秒）
    time_use2 = (end2.tv_sec - start2.tv_sec) * 1000.0 +
               (end2.tv_usec - start2.tv_usec) / 1000.0;
    printf("inference_yolov8_model rknpu2 耗时：%.3f ms\n", time_use2);

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    printf("rknn_outputs_get\n");
    // 开始计时
    gettimeofday(&start3, NULL);
    //获得输出
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        //goto out;
    }else if (ret == 0)
    {
        printf("rknn_outputs_get success! ret=%d\n", ret);
    }
    // 结束计时
    gettimeofday(&end3, NULL);

    // 计算耗时（微秒→毫秒）
    double time_use3 = (end3.tv_sec - start3.tv_sec) * 1000.0 +
               (end3.tv_usec - start3.tv_usec) / 1000.0;
    printf("rknn_outputs_get 耗时：%.3f ms\n", time_use3);

    // Post Process
    //如果rk平台的后处理有问题的话可以试试换成联咏的后处理流程
    // 开始计时
    gettimeofday(&start4, NULL);
    post_process_yolo26(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
    printf("post_process success!\n");
    // 结束计时
    gettimeofday(&end4, NULL);

    // 计算耗时（微秒→毫秒）
    double time_use4 = (end4.tv_sec - start4.tv_sec) * 1000.0 +
               (end4.tv_usec - start4.tv_usec) / 1000.0;
    printf("post_process 耗时：%.3f ms\n", time_use4);
    //printf("**********inference_yolov8_model rknpu2 耗时：%.3f ms\n", time_use2);
    // Remeber to release rknn output
    // 开始计时
    gettimeofday(&start5, NULL);
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    // 结束计时
    gettimeofday(&end5, NULL);

    // 计算耗时（微秒→毫秒）
    double time_use5 = (end5.tv_sec - start5.tv_sec) * 1000.0 +
               (end5.tv_usec - start5.tv_usec) / 1000.0;
    printf("rknn_outputs_release 耗时：%.3f ms\n", time_use5);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}