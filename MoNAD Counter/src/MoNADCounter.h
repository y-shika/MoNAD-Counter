#ifndef __MONADCOUNTER_H_INCLUDED__
#define __MONADCOUNTER_H_INCLUDED__

#include "csvFile.h"

#include <random>

// �W�����͂̃G���[�����ɗp����.
#include <limits>

class MoNAD : public csvFile
{
private:
	// �f�q�Əd�݂̍s��v�Z�ɂ�����, Eigen�f�t�H���g�̗�x�N�g���̂܂܂���, �w�f�q * �d��([2. 1] * [2, 2])�x�̌`�Ōv�Z�ł��Ȃ�����, �s�x�N�g����p����.
	// (�]�u��p���Ď������ł��邪, �R�[�h�̉ǐ����ۂ���Ȃ�)
	Eigen::RowVectorXd in;
	Eigen::RowVectorXd ssu;
	Eigen::RowVectorXd bl;
	Eigen::RowVectorXd hid;
	Eigen::RowVectorXd out;
	Eigen::RowVectorXd rl;

	Eigen::MatrixXd w_in_hid;
	Eigen::MatrixXd w_ssu_hid;
	Eigen::MatrixXd w_bl_hid;
	Eigen::MatrixXd w_hid_out;
	Eigen::MatrixXd w_hid_rl;

	Eigen::RowVectorXd theta_hid;
	Eigen::RowVectorXd theta_out;
	Eigen::RowVectorXd theta_rl;

	double alpha;
	double error_threshold;

	double p;
	double errorTotal_out;

	Eigen::RowVectorXd sigmoid(Eigen::RowVectorXd vec);

	void setData(int i);

	void forwardPropagation();
	void backPropagation(int i);

	void saveLog_w_theta();
	void saveLog_p_error();

	double calc_error(int i);

	void calc_p_errorTotal();

	void inputCommand();
	void feedback_ssu_bl();

public:
	std::vector<Eigen::MatrixXd> w_in_hid_log;
	std::vector<Eigen::MatrixXd> w_ssu_hid_log;
	std::vector<Eigen::MatrixXd> w_bl_hid_log;
	std::vector<Eigen::MatrixXd> w_hid_out_log;
	std::vector<Eigen::MatrixXd> w_hid_rl_log;

	std::vector<Eigen::RowVectorXd> theta_hid_log;
	std::vector<Eigen::RowVectorXd> theta_out_log;
	std::vector<Eigen::RowVectorXd> theta_rl_log;

	std::vector<double> p_log;
	std::vector<double> errorTotal_out_log;

	MoNAD();
	//~MoNAD();

	void learn();
	void counter();
};

#endif

