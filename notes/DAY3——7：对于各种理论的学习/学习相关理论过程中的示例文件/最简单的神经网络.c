#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// ---------- 固定参数 ----------
#define SAMPLE_NUM 100    // 训练数据个数（-3 到 3 取100个点）
#define HIDDEN_SIZE 10    // 隐藏层神经元个数
#define EPOCHS 3000       // 训练轮数
#define LR 0.01           // 学习率（微调步长）

int main() {
    // ---------- 1. 准备训练数据 ----------
    double X[SAMPLE_NUM][1];     // 输入 x
    double Y[SAMPLE_NUM][1];     // 真实值 sin(x)
    SetConsoleOutputCP(65001);
    for (int i = 0; i < SAMPLE_NUM; i++) {
        X[i][0] = -3.0 + (6.0 * i) / (SAMPLE_NUM - 1);  // 从 -3 均匀到 3
        Y[i][0] = sin(X[i][0]);                         // 标准答案
    }

    // ---------- 2. 初始化那4组数字（随机数） ----------
    srand(42); // 固定随机种子，让你我结果一致

    // 组A (W1): 1行 x 10列
    double W1[1][HIDDEN_SIZE];
    // 组B (b1): 1行 x 10列
    double b1[1][HIDDEN_SIZE];
    // 组C (W2): 10行 x 1列
    double W2[HIDDEN_SIZE][1];
    // 组D (b2): 1行 x 1列
    double b2[1][1];

    // 把随机数填进去（范围 -1 到 1）
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        W1[0][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        b1[0][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        W2[j][0] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    b2[0][0] = ((double)rand() / RAND_MAX) * 2 - 1;

    printf("开始训练...\n");
    printf("初始时，W1 的第一个数字是: %.4f\n\n", W1[0][0]);

    // ---------- 3. 开始训练（重复微调3000轮） ----------
    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        // ========== 正向传播（前向计算） ==========
        // 隐藏层线性结果: hidden_linear[i][j] = X[i][0] * W1[0][j] + b1[0][j]
        double hidden_linear[SAMPLE_NUM][HIDDEN_SIZE];
        // 隐藏层激活后: hidden_activated[i][j] = tanh(hidden_linear[i][j])
        double hidden_activated[SAMPLE_NUM][HIDDEN_SIZE];
        // 预测结果: Y_pred[i][0] = sum( hidden_activated[i][j] * W2[j][0] ) + b2[0][0]
        double Y_pred[SAMPLE_NUM][1];

        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                hidden_linear[i][j] = X[i][0] * W1[0][j] + b1[0][j];
                hidden_activated[i][j] = tanh(hidden_linear[i][j]);
            }
            // 计算输出层（把10个神经元的结果加权求和）
            double sum = 0.0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += hidden_activated[i][j] * W2[j][0];
            }
            Y_pred[i][0] = sum + b2[0][0];
        }

        // ========== 计算损失（猜得有多离谱） ==========
        double loss = 0.0;
        for (int i = 0; i < SAMPLE_NUM; i++) {
            double diff = Y_pred[i][0] - Y[i][0];
            loss += diff * diff;
        }
        loss = loss / SAMPLE_NUM; // 均方误差

        // ========== 反向传播（手动计算每个数字该怎么微调） ==========
        // 第1步：输出层误差敏感度 dLoss_dYpred[i] = 2*(预测 - 真实)/样本数
        double dLoss_dYpred[SAMPLE_NUM][1];
        for (int i = 0; i < SAMPLE_NUM; i++) {
            dLoss_dYpred[i][0] = 2 * (Y_pred[i][0] - Y[i][0]) / SAMPLE_NUM;
        }

        // 第2步：计算组C (W2) 和 组D (b2) 的梯度
        double dW2[HIDDEN_SIZE][1] = {0};
        double db2[1][1] = {0};
        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                dW2[j][0] += hidden_activated[i][j] * dLoss_dYpred[i][0];
            }
            db2[0][0] += dLoss_dYpred[i][0];
        }

        // 第3步：把误差往后传，算隐藏层的误差
        double dLoss_dHidden[SAMPLE_NUM][HIDDEN_SIZE];
        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                dLoss_dHidden[i][j] = dLoss_dYpred[i][0] * W2[j][0];
            }
        }

        // 第4步：tanh 的导数 = 1 - 激活值^2
        double dLoss_dHidden_linear[SAMPLE_NUM][HIDDEN_SIZE];
        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double dtanh = 1 - hidden_activated[i][j] * hidden_activated[i][j];
                dLoss_dHidden_linear[i][j] = dLoss_dHidden[i][j] * dtanh;
            }
        }

        // 第5步：计算组A (W1) 和 组B (b1) 的梯度
        double dW1[1][HIDDEN_SIZE] = {0};
        double db1[1][HIDDEN_SIZE] = {0};
        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                dW1[0][j] += X[i][0] * dLoss_dHidden_linear[i][j];
                db1[0][j] += dLoss_dHidden_linear[i][j];
            }
        }

        // ========== 梯度下降（真正动手微调这4组数字） ==========
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W1[0][j] -= LR * dW1[0][j];
            b1[0][j] -= LR * db1[0][j];
            W2[j][0] -= LR * dW2[j][0];
        }
        b2[0][0] -= LR * db2[0][0];

        // ========== 每500轮打印一次进度 ==========
        if (epoch % 500 == 0) {
            printf("----- 第 %d 轮 -----\n", epoch);
            printf("损失 Loss: %.6f\n", loss);
            printf("此时 W1 的第一个数字变成了: %.4f\n", W1[0][0]);
            printf("此时 W2 的第一个数字变成了: %.4f\n\n", W2[0][0]);
        }
    }

    printf("训练结束！\n\n");

    // ---------- 4. 把最终结果写入文件（方便你用 Excel 画图） ----------
    FILE *fp = fopen("result.csv", "w");
    if (fp == NULL) {
        printf("无法创建文件！\n");
        return 1;
    }
    fprintf(fp, "x,真实sin(x),MLP预测值\n");
    
    // 为了画图更平滑，在 -3 到 3 之间取 300 个点做测试
    int test_num = 300;
    for (int i = 0; i < test_num; i++) {
        double x = -3.0 + (6.0 * i) / (test_num - 1);
        
        // 用训练好的最终数字计算预测值
        double hidden_out[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_out[j] = tanh(x * W1[0][j] + b1[0][j]);
        }
        double y_pred = b2[0][0];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            y_pred += hidden_out[j] * W2[j][0];
        }
        double y_true = sin(x);
        fprintf(fp, "%.6f,%.6f,%.6f\n", x, y_true, y_pred);
    }
    fclose(fp);

    printf("结果已写入 result.csv 文件。\n");
    printf("用 Excel 打开它，插入折线图，就能看到蓝线（预测）和红线（真实）重合了！\n");

    return 0;
}