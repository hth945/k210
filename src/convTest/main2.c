#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "encoding.h"
#include "fpioa.h"
#include "gpiohs.h"
#include "kpu.h"
#include "plic.h"
#include "sleep.h"
#include "sysctl.h"
#include "uart.h"
#include "uarths.h"
#include "utils.h"

//激活函数折点表，设置为y=x，即直接输出卷积结果
// y=(uint8_t)((((uint64_t)(x - x_start) * y_mul) >> shift) + bias);

kpu_activate_table_t active_addr __attribute__((aligned(256))) =
    {.activate_para =
         {// x =36bit
          {.data = {.shift_number = 0, .y_mul = 0, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}}},
     .activate_para_bias0.data = {.result_bias = {0, 0, 0, 0, 0, 0, 0, 0}},
     .activate_para_bias1.data = {.result_bias = {0, 0, 0, 0, 0, 0, 0, 0}}};

// y = (x*norm_mul)>>norm_shift + norm_add
kpu_batchnorm_argument_t bwsx_base_addr[] __attribute__((aligned(128))) = {
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
};

//卷积参数

kpu_task_t task;
kpu_layer_argument_t la __attribute__((aligned(128)));
uint16_t conv_data_u16[1 * 3 * 3]
    __attribute__((aligned(128))) = {0, 0, 0, 0, 0, 0, 0, 0, 0};

// uint8_t img_src[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
// uint8_t img_dst[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
#define size 64
uint8_t img_src[size * size] __attribute__((aligned(128))) = {0};
uint8_t img_dst[size * size] __attribute__((aligned(128))) = {0};

int flag = 0;
// plic_irq_callback_t
static int kpu_done(void *ctx) {
  flag = 1;
  // char *hel = {"OK!\n"};
  // uart_send_data(UART_DEVICE_3, hel, strlen(hel));
  // char str[32];
  // for (int i = 0; i < sizeof(img_dst); i++) {
  //   sprintf(str, "%d, ", img_dst[i]);
  //   uart_send_data(UART_DEVICE_3, str, strlen(hel));
  // }
  uart_send_data(UART_DEVICE_3, img_dst, sizeof(img_dst));
  return 0;
}

int main(void) {
#define PLL0_OUTPUT_FREQ 600000000UL
#define PLL1_OUTPUT_FREQ 400000000UL
#define PLL2_OUTPUT_FREQ 45158400UL

  sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
  sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
  sysctl_pll_set_freq(SYSCTL_PLL2, PLL2_OUTPUT_FREQ);
  sysctl_clock_enable(SYSCTL_CLOCK_AI);

  /********串口初始化************/
  fpioa_set_function(10, FUNC_UART3_RX);
  fpioa_set_function(9, FUNC_UART3_TX);
  plic_init();
  uart_init(UART_DEVICE_3);
  uart_configure(UART_DEVICE_3, 115200, 8, UART_STOP_1, UART_PARITY_NONE);

  /********数据初始化************/
  memset(img_src, 1, sizeof(img_src));
  memset(conv_data_u16, 0, sizeof(conv_data_u16));
  // img_src[4] = 1;
  // conv_data_u16[4] = 1;
  for (int i = 0; i < 9; i++) {
    conv_data_u16[i] = 1;
  }
  // uart_send_data(UART_DEVICE_3, img_src, sizeof(img_src));

  /*******网络初始化************/
  uint16_t w = size;
  uint16_t h = size;
  uint8_t ch_in = 1;
  uint8_t ch_out = 1;

  la.conv_value.data.arg_x = 0;
  la.conv_value.data.shr_x = 0;
  la.conv_value.data.arg_w = 0;
  la.conv_value.data.shr_w = 0;
  la.conv_value2.data.arg_add = 0;

  la.conv_value.data.arg_x = 0;
  la.conv_value.data.shr_x = 0;
  for (int i = 1; i < 16; i++) {
    active_addr.activate_para[i].data.shift_number = 0;
    active_addr.activate_para[i].data.y_mul = 1;
    active_addr.activate_para[i].data.x_start = 0;
  }

  la.kernel_offset.data.coef_row_offset = 0;     //固定为0
  la.kernel_offset.data.coef_column_offset = 0;  //固定为0
  //激活函数配置-
  la.kernel_calc_type_cfg.data.load_act = 1;  //使能激活函数
  la.kernel_calc_type_cfg.data.active_addr = (uint64_t)&active_addr;
  //初始化激活表
  // row_switch_addr = math.ceil(i_row_wid / 64)
  // channel_switch_addr = i_col_high * row_switch_addr
  la.kernel_calc_type_cfg.data.row_switch_addr =
      (w + 63) / 64;  //图像宽度占用的单元数
  la.kernel_calc_type_cfg.data.channel_switch_addr = (w + 63) / 64 * h;
  la.kernel_calc_type_cfg.data.coef_size = 0;  //固定为0
  la.kernel_calc_type_cfg.data.coef_group = 1;

  //中断设置--
  la.interrupt_enabe.data.depth_wise_layer = 0;  //常规卷积层
  la.interrupt_enabe.data.int_en = 0;            //使能中断
  la.interrupt_enabe.data.full_add = 0;          //??
  la.interrupt_enabe.data.ram_flag = 0;          //??
  // dma设置，知道是输出数据使用的DMA--
  la.dma_parameter.data.dma_total_byte =
      w * h * ch_out - 1;                   //总共的DMA传输数量
  la.dma_parameter.data.send_data_out = 1;  //使能数据的dma输出
  la.dma_parameter.data.channel_byte_num = w * h - 1;  //单通道的DMA传输数量
  //卷积运算参数设置--
  // arg_x 为24bit,shr_x 为4bit, 在conv_float2u16中设置

  //写回设置--
  la.write_back_cfg.data.wb_row_switch_addr = (w + 63) / 64;  // ceil(16/64)=1
  la.write_back_cfg.data.wb_channel_switch_addr = (w + 63) / 64 * h;  // 16*1
  la.write_back_cfg.data.wb_group = 1;                                // 64/w
  //图像尺寸设置--
  la.image_size.data.i_row_wid = w - 1;  //输入长宽
  la.image_size.data.i_col_high = h - 1;
  la.image_size.data.o_row_wid = w - 1;  //输出长宽
  la.image_size.data.o_col_high = h - 1;
  //池化类型设置-
  la.kernel_pool_type_cfg.data.bypass_conv = 0;  //不略过卷积
  la.kernel_pool_type_cfg.data.pad_value = 0x0;  //边界填充0
  la.kernel_pool_type_cfg.data.load_para = 1;    //允许归一化
  la.kernel_pool_type_cfg.data.pad_type = 0;     //使用填充值
  la.kernel_pool_type_cfg.data.kernel_type = 1;  // 3x3
  la.kernel_pool_type_cfg.data.pool_type = 0;    //池化类型，跳过
  la.kernel_pool_type_cfg.data.dma_burst_size = 15;  // dma突发传送大小，16字节
  la.kernel_pool_type_cfg.data.bwsx_base_addr = (uint64_t)&bwsx_base_addr;
  //批归一化首地址
  la.kernel_pool_type_cfg.data.first_stride =
      h < 256 ? 0 : 1;  //图像高度未超过255
  //图像通道设置--
  la.image_channel_num.data.o_ch_num_coef =
      ch_out - 1;  //一次性参数加载可计算的通道数
  la.image_channel_num.data.i_ch_num = ch_in - 1;   //输入通道
  la.image_channel_num.data.o_ch_num = ch_out - 1;  //输出通道
  //卷积参数设置-
  la.kernel_load_cfg.data.load_time = 0;  //卷积加载次数，不超过72KB，只加载一次
  la.kernel_load_cfg.data.para_size = 2 * 9 * ch_in * ch_out;  //卷积参数大小
  la.kernel_load_cfg.data.para_start_addr = (uint64_t)conv_data_u16;
  //起始地址
  la.kernel_load_cfg.data.load_coor = 1;  //允许加载卷积参数
  //计算地址设置--
  la.image_addr.data.image_src_addr = (uint64_t)0x0;  //一个为0
  la.image_addr.data.image_dst_addr =
      (uint64_t)(0x200000 / 64 - (w + 63) / 64 * h * ch_out);

  /* init kpu task*/
  task.layers = &la;
  task.layers_length = 1;   //单层
  task.eight_bit_mode = 0;  // 16bit模式
  task.output_scale = 1.0;  //输出的缩放
  task.output_bias = 0;     //输出的偏置

  /* start to calculate */
  kpu_run(&task, DMAC_CHANNEL5, img_src, img_dst, kpu_done);

  while (1) {
    // char *hel = {"hello world!\n"};
    // char str[32] = {0};
    // sprintf(str, "%d, \n", flag);
    // uart_send_data(UART_DEVICE_3, str, strlen(str));
    // int n = strlen(str);
    // sprintf(str, "%d, \n", n);
    // uart_send_data(UART_DEVICE_3, str, 4);
    msleep(1);
    // uart_send_data(UART_DEVICE_3, hel, strlen(hel));
    msleep(1000);
    // uart_send_data(UART_DEVICE_3, img_dst, sizeof(img_dst));
    // for (int i = 0; i < sizeof(img_dst); i++) {
    //   sprintf(str, "%d, ", img_dst[i]);
    //   uart_send_data(UART_DEVICE_3, str, strlen(hel));
    // }
  }
}
