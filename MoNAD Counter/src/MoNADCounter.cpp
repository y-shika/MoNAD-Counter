#include "MoNADCounter.h"

MoNAD::MoNAD() {
	in.resize(INNUM);
	ssu.resize(SSUNUM);
	bl.resize(BLNUM);
	hid.resize(HIDNUM);
	out.resize(OUTNUM);
	rl.resize(RLNUM);

	w_in_hid.resize(INNUM, HIDNUM);
	w_ssu_hid.resize(SSUNUM, HIDNUM);
	w_bl_hid.resize(BLNUM, HIDNUM);
	w_hid_out.resize(HIDNUM, OUTNUM);
	w_hid_rl.resize(HIDNUM, RLNUM);

	theta_hid.resize(HIDNUM);
	theta_out.resize(OUTNUM);
	theta_rl.resize(RLNUM);

	// 乱数生成器 [-1.0, 1.0)
	std::random_device rnd;
	std::mt19937_64 mt(rnd());
	std::uniform_real_distribution<> rand100(-1.0, 1.0);

	for (int i_in = 0; i_in < INNUM; i_in++)
		for (int i_hid = 0; i_hid < HIDNUM; i_hid++)
			w_in_hid(i_in, i_hid) = rand100(mt);

	for (int i_ssu = 0; i_ssu < SSUNUM; i_ssu++)
		for (int i_hid = 0; i_hid < HIDNUM; i_hid++)
			w_ssu_hid(i_ssu, i_hid) = rand100(mt);

	for (int i_bl = 0; i_bl < BLNUM; i_bl++)
		for (int i_hid = 0; i_hid < HIDNUM; i_hid++)
			w_bl_hid(i_bl, i_hid) = rand100(mt);

	for (int i_hid = 0; i_hid < HIDNUM; i_hid++)
		for (int i_out = 0; i_out < OUTNUM; i_out++)
			w_hid_out(i_hid, i_out) = rand100(mt);

	for (int i_hid = 0; i_hid < HIDNUM; i_hid++)
		for (int i_rl = 0; i_rl < RLNUM; i_rl++)
			w_hid_rl(i_hid, i_rl) = rand100(mt);

	for (int i_hid = 0; i_hid < HIDNUM; i_hid++) theta_hid[i_hid] = rand100(mt);
	for (int i_out = 0; i_out < OUTNUM; i_out++) theta_out[i_out] = rand100(mt);
	for (int i_rl = 0; i_rl < RLNUM; i_rl++) theta_rl[i_rl] = rand100(mt);

	alpha = 0.05;
	error_threshold = 0.0005;

	p = 0;
	errorTotal_out = 0;

	read();
}

void MoNAD::learn() {
	bool flagLearned = false;
	double error_out = 0;
	int learnCount = 0;

	while (!flagLearned) {
		for (int i = 0; i < Teach_in.size(); i++) {
			setData(i);
			forwardPropagation();
			error_out = calc_error(i);

			// バックプロパゲーションによる学習
			if (error_out >= error_threshold) {
				while (error_out >= error_threshold) {
					backPropagation(i);
					setData(i);
					forwardPropagation();
					error_out = calc_error(i);
				}
				learnCount++;
			}
		}
		calc_p_errorTotal();
		std::cout << learnCount << ", " << p << ", " << errorTotal_out << std::endl;

		if (p == 100 && errorTotal_out < 0.01) flagLearned = true;
	}

	std::cout << "学習が完了しました." << std::endl;
}

void MoNAD::counter() {
	ssu << 0, 0;
	bl << 1, 1;

	while (1) {
		inputCommand();

		forwardPropagation();

		// Eigen行列をそのまま標準出力に出すと, 先頭に空白文字が入ることがあるため, 配列のようにして出している.
		// TODO : 出力素子が増えたときのために, 行列の先頭に空白文字が入らないようにする.
		std::cout << out[0] << "  " << out[1] << std::endl;

		feedback_ssu_bl();
	}
}

void MoNAD::inputCommand() {
	// 入力素子数に合わせて汎用性高めるなら, 標準入力からの処理にも一工夫加えたいが,
	// この関数自体が今回のカウンター特化な性質なため, 目を瞑ることとした.
	std::cout << "in : ";
	for (std::cin >> in[0]; !std::cin; std::cin >> in[0]) {
		std::cin.clear();

		// ignoreに引数を与えないと, 数字以外の複数文字を入れた場合にその文字数だけメッセージが出力されるため, このようにした.
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::cout << "入力が間違っています." << std::endl << "in : ";
	}
}

void MoNAD::feedback_ssu_bl() {
	ssu = out;
	bl = rl;
}

double MoNAD::calc_error(int i) {
	double error_out = 0;

	error_out = (out - Teach_out[i]).cwiseAbs2().sum();
	return error_out;
}

void MoNAD::calc_p_errorTotal() {
	int correctCount = 0;
	double error_out = 0;
	errorTotal_out = 0;

	for (int i = 0; i < Teach_in.size(); i++) {
		setData(i);
		forwardPropagation();
		error_out = calc_error(i);
		errorTotal_out = errorTotal_out + error_out;

		if (error_out < error_threshold) correctCount++;
	}

	p = correctCount * 100 / Teach_out.size();

	saveLog_p_error();
}

Eigen::RowVectorXd MoNAD::sigmoid(Eigen::RowVectorXd vec) {
	Eigen::RowVectorXd _vec;

	_vec = 1.0 / (1.0 + exp(-vec.array()));
	return _vec;
}

void MoNAD::setData(int i) {
	in = Teach_in[i];
	ssu = Teach_ssu[i];
	bl = Teach_bl[i];
}

void MoNAD::forwardPropagation() {
	hid = sigmoid(in * w_in_hid + ssu * w_ssu_hid + bl * w_bl_hid - theta_hid);
	out = sigmoid(hid * w_hid_out - theta_out);
	rl = sigmoid(hid * w_hid_rl - theta_rl);
}

void MoNAD::backPropagation(int i) {
	Eigen::Matrix<double, 1, OUTNUM> delta_out;
	Eigen::Matrix<double, 1, RLNUM> delta_rl;
	Eigen::Matrix<double, 1, HIDNUM> delta_hid;

	Eigen::Matrix<double, INNUM, HIDNUM> delta_w_in_hid;
	Eigen::Matrix<double, SSUNUM, HIDNUM> delta_w_ssu_hid;
	Eigen::Matrix<double, BLNUM, HIDNUM> delta_w_bl_hid;
	Eigen::Matrix<double, HIDNUM, OUTNUM> delta_w_hid_out;
	Eigen::Matrix<double, HIDNUM, RLNUM> delta_w_hid_rl;

	Eigen::Matrix<double, 1, HIDNUM> delta_theta_hid;
	Eigen::Matrix<double, 1, OUTNUM> delta_theta_out;
	Eigen::Matrix<double, 1, RLNUM> delta_theta_rl;

	delta_out = (out - Teach_out[i]).array() * (1 - out.array()).array() * out.array();
	delta_rl = (rl - Teach_rl[i]).array() * (1 - rl.array()).array() * out.array();
	delta_hid = (w_hid_out * delta_out.transpose() + w_hid_rl * delta_rl.transpose()).transpose().array() * (1 - hid.array()).array() * hid.array();

	delta_w_in_hid = alpha * (delta_hid.transpose() * in).transpose().array();
	delta_w_ssu_hid = alpha * (delta_hid.transpose() * ssu).transpose().array();
	delta_w_bl_hid = alpha * (delta_hid.transpose() * bl).transpose().array();
	delta_w_hid_out = alpha * (delta_out.transpose() * hid).transpose().array();
	delta_w_hid_rl = alpha * (delta_rl.transpose() * hid).transpose().array();

	delta_theta_hid = -alpha * delta_hid.array();
	delta_theta_out = -alpha * delta_out.array();
	delta_theta_rl = -alpha * delta_rl.array();

	w_in_hid = w_in_hid - delta_w_in_hid;
	w_ssu_hid = w_ssu_hid - delta_w_ssu_hid;
	w_bl_hid = w_bl_hid - delta_w_bl_hid;
	w_hid_out = w_hid_out - delta_w_hid_out;
	w_hid_rl = w_hid_rl - delta_w_hid_rl;

	theta_hid = theta_hid - delta_theta_hid;
	theta_out = theta_out - delta_theta_out;
	theta_rl = theta_rl - delta_theta_rl;

	saveLog_w_theta();
}

void MoNAD::saveLog_w_theta() {
	w_in_hid_log.push_back(w_in_hid);
	w_ssu_hid_log.push_back(w_ssu_hid);
	w_bl_hid_log.push_back(w_bl_hid);
	w_hid_out_log.push_back(w_hid_out);
	w_hid_rl_log.push_back(w_hid_rl);

	theta_hid_log.push_back(theta_hid);
	theta_out_log.push_back(theta_out);
	theta_rl_log.push_back(theta_rl);
}

void MoNAD::saveLog_p_error() {
	p_log.push_back(p);
	errorTotal_out_log.push_back(errorTotal_out);
}