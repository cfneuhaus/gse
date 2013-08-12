#ifndef GSE_EIGEN_H
#define GSE_EIGEN_H

//#ifndef EIGEN_DONT_ALIGN_STATICALLY
//#define EIGEN_DONT_ALIGN_STATICALLY
//#endif
#include "GSEDefs.h"
#include <unsupported/Eigen/AutoDiff>

GSE_NS_BEGIN

using namespace Eigen;

typedef Matrix<double, 10, 1>    Vector10d;
typedef Matrix<double, 10, 10>    Matrix10d;
typedef Matrix<double, 1, 1>    Vector1d;
typedef Matrix<double, 7, 1>    Vector7d;
typedef Matrix<double, 7, 7>    Matrix7d;
typedef Matrix<double, 6, 1>    Vector6d;
typedef Matrix<double, 6, 6>    Matrix6d;

// Some Autodiff workarounds, since the Eigen module is buggy
// See also my bug report: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=234

template<typename T>
AutoDiffScalar<T> operator-(const AutoDiffScalar<T>& v)
{
	return (-1)*v;
}

template<typename T>
bool operator<(const AutoDiffScalar<T>& a, const AutoDiffScalar<T>& b)
{
	return a.value()<b.value();
}

#define EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(FUNC,CODE) \
  template<typename DerType> \
  inline const Eigen::AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_multiple_op<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar>, const typename Eigen::internal::remove_all<DerType>::type> > \
  FUNC(const Eigen::AutoDiffScalar<DerType>& x) { \
	using namespace Eigen; \
	typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar; \
	typedef AutoDiffScalar<CwiseUnaryOp<Eigen::internal::scalar_multiple_op<Scalar>, const typename Eigen::internal::remove_all<DerType>::type> > ReturnType; \
	CODE; \
  }

GSE_NS_END

namespace std{

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(atan,
  return ReturnType(std::atan(x.value()),x.derivatives() * (Scalar(1.0)/(x.value()*x.value()+Scalar(1.0))));)
EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(acos,
  return ReturnType(std::acos(x.value()),x.derivatives() * (Scalar(-1.0)/(std::sqrt(Scalar(1.0)-x.value()*x.value()))));)
}
#undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY

#endif // EIGEN_H
