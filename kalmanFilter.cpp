#include "kalmanFilter.h"
#include <Eigen/Cholesky>

namespace byte_kalman
{
	const double KalmanFilter::chi2inv95[10] = {
	0,
	3.8415,
	5.9915,
	7.8147,
	9.4877,
	11.070,
	12.592,
	14.067,
	15.507,
	16.919
	};
	KalmanFilter::KalmanFilter()
	{
		int ndim = 4;
		double dt = 1.;

		_motion_mat = Eigen::MatrixXf::Identity(8, 8);
		for (int i = 0; i < ndim; i++) {
			_motion_mat(i, ndim + i) = dt;
		}
		_update_mat = Eigen::MatrixXf::Identity(4, 8);

		this->_std_weight_position = 1. / 20;
		this->_std_weight_velocity = 1. / 160;
	}

	KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement)
	{
		DETECTBOX mean_pos = measurement;
		DETECTBOX mean_vel;
		for (int i = 0; i < 4; i++) mean_vel(i) = 0;

		KAL_MEAN mean;
		for (int i = 0; i < 8; i++) {
			if (i < 4) mean(i) = mean_pos(i);
			else mean(i) = mean_vel(i - 4);
		}

		KAL_MEAN std;
		std(0) = 2 * _std_weight_position * measurement[3];
		std(1) = 2 * _std_weight_position * measurement[3];
		std(2) = 1e-2;
		std(3) = 2 * _std_weight_position * measurement[3];
		std(4) = 10 * _std_weight_velocity * measurement[3];
		std(5) = 10 * _std_weight_velocity * measurement[3];
		std(6) = 1e-5;
		std(7) = 10 * _std_weight_velocity * measurement[3];

		KAL_MEAN tmp = std.array().square();
		KAL_COVA var = tmp.asDiagonal();
		return std::make_pair(mean, var);
	}

	void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
	{
		//revise the data;
		//位置噪声的标准差
		DETECTBOX std_pos;
		std_pos << _std_weight_position * mean(3),
			_std_weight_position * mean(3),
			1e-2,
			_std_weight_position * mean(3);
		//速度噪声的标准差
		DETECTBOX std_vel;
		std_vel << _std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			1e-5,
			_std_weight_velocity * mean(3);
		//运动噪声的协方差矩阵
		KAL_MEAN tmp;
		tmp.block<1, 4>(0, 0) = std_pos;
		tmp.block<1, 4>(0, 4) = std_vel;
		tmp = tmp.array().square();
		KAL_COVA motion_cov = tmp.asDiagonal();
		//预测均值
		KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
		//预测协方差
		KAL_COVA covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
		covariance1 += motion_cov;

		mean = mean1;
		covariance = covariance1;
	}

	//状态空间 → 观测空间,把内部状态向量 mean（例如 [cx, cy, a, h, vx, vy, va, vh]）映射成检测框的形式（观测空间） [cx, cy, a, h]。
	KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance)
	{
		DETECTBOX std;
		std << _std_weight_position * mean(3), _std_weight_position * mean(3),
			1e-1, _std_weight_position * mean(3);
		KAL_HMEAN mean1 = _update_mat * mean.transpose();
		KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
		Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
		diag = diag.array().square().matrix();
		covariance1 += diag;
		//    covariance1.diagonal() << diag;
		return std::make_pair(mean1, covariance1);
	}

	//根据检测框 measurement（观测值）来修正预测值。
	KAL_DATA
		KalmanFilter::update(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const DETECTBOX &measurement)
	{
		KAL_HDATA pa = project(mean, covariance);
		KAL_HMEAN projected_mean = pa.first;
		KAL_HCOVA projected_cov = pa.second;

		//chol_factor, lower =
		//scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
		//kalmain_gain =
		//scipy.linalg.cho_solve((cho_factor, lower),
		//np.dot(covariance, self._upadte_mat.T).T,
		//check_finite=False).T
		Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
		Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
		Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
		auto tmp = innovation * (kalman_gain.transpose());
		KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
		KAL_COVA new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
		return std::make_pair(new_mean, new_covariance);
	}

	//计算“预测轨迹”与“检测框”之间的马氏距离（Mahalanobis distance），用来衡量匹配的合理性（即预测和检测是否属于同一目标）。
    //注意！！！ 这个函数的作用是计算 马氏距离 (Mahalanobis distance)，用于判断某个“观测值（测量值）”是否与当前卡尔曼预测状态匹配
    //在 多目标跟踪（比如 ByteTrack、SORT、DeepSORT）中，卡尔曼滤波不仅预测单个目标的下一个位置，还要在每一帧中决定“哪个检测框对应哪个目标轨迹”因此，需要计算一个“距离矩阵”来匹配哪个轨迹与哪个检测框
    //在单目标中没有检测框匹配，所以不需要马氏距离来判断哪个检测属于哪个轨迹。
    //在单目标追踪中，经常出现误检测的情况，所以应当在更新运动矩阵和协方差矩阵前做一次马氏距离匹配
 	Eigen::Matrix<float, 1, -1>
		KalmanFilter::gating_distance(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const std::vector<DETECTBOX> &measurements,
			bool only_position)
	{
		KAL_HDATA pa = this->project(mean, covariance);
		if (only_position) {
			printf("not implement!");
			exit(0);
		}
		KAL_HMEAN mean1 = pa.first;
		KAL_HCOVA covariance1 = pa.second;

		//    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
		DETECTBOXSS d(measurements.size(), 4);
		int pos = 0;
		for (DETECTBOX box : measurements) {
			d.row(pos++) = box - mean1;
		}
		Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
		Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
		auto zz = ((z.array())*(z.array())).matrix();
		auto square_maha = zz.colwise().sum();
		return square_maha;
	}
}