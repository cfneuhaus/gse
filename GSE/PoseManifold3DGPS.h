#ifndef POSEMANIFOLD3DGPS_H
#define POSEMANIFOLD3DGPS_H

#include "eigen.h"

template<typename Scalar_>
struct PoseManifold3DGPS
{
	template<typename T>
	struct Other
	{
		typedef PoseManifold3DGPS<T> Type;
	};

	enum {PoseDim=10, DeltaDim=9 };
	typedef Scalar_ Scalar;
	typedef Matrix<Scalar,PoseDim,1> PoseVec;
	typedef Matrix<Scalar,PoseDim,PoseDim> PoseMat; // maybe not needed
	typedef Matrix<Scalar,DeltaDim,1> DeltaVec;
	typedef Matrix<Scalar,DeltaDim,DeltaDim> DeltaMat;

	static PoseVec zeroPose() { PoseVec ret; ret<< 0,0,0,1,0,0,0,0,0,0; return ret; }

#define Tsegment template segment
	static PoseVec manifoldAdd(const PoseVec& x, const DeltaVec& d)
	{
		TQuat qx(x(3),x(4),x(5),x(6));
		TQuat qnew=qx*scaledAxisToQuat(d.Tsegment<3>(3));
		//RaRd
		//Ratd+ta

		PoseVec ret;
		ret.Tsegment<3>(0)=d.Tsegment<3>(0)+x.Tsegment<3>(0);
		ret(3)=qnew.w();	ret(4)=qnew.x();	ret(5)=qnew.y();	ret(6)=qnew.z();

		// gps
		ret.Tsegment<3>(7)+=d.Tsegment<3>(6);
		return ret;
	}

	static DeltaVec manifoldSub(const PoseVec& x, const PoseVec& y)
	{
		//RyTRx
		//RyT(x-y)
		TQuat qx(x(3),x(4),x(5),x(6));
		TQuat qy(y(3),y(4),y(5),y(6));
		TQuat qy_conj(y(3),-1.0*y(4),-1.0*y(5),-1.0*y(6));
		TQuat qres=qy_conj*qx;

		DeltaVec ret;
		ret.Tsegment<3>(0)=(x.Tsegment<3>(0)-y.Tsegment<3>(0));
		ret.Tsegment<3>(3)=quatToScaledAxis(qres);

		// gps
		ret.Tsegment<3>(6)=x.Tsegment<3>(7)-y.Tsegment<3>(7);
		return ret;
	}
	static PoseVec poseMinus(const PoseVec& x, const PoseVec& y)
	{
		TQuat qx(x(3),x(4),x(5),x(6));
		TQuat qy(y(3),y(4),y(5),y(6));
		TQuat qy_conj(y(3),-1.0*y(4),-1.0*y(5),-1.0*y(6));
		TQuat qres=qy_conj*qx;
		PoseVec ret;
		ret.Tsegment<3>(0)=rotatePt(qy_conj,x.Tsegment<3>(0)-y.Tsegment<3>(0));
		ret(3)=qres.w();	ret(4)=qres.x();	ret(5)=qres.y();	ret(6)=qres.z();

		// gps
		ret.Tsegment<3>(7)=x.Tsegment<3>(7)-y.Tsegment<3>(7);
		return ret;
	}
	// not for actual optimization
	static PoseVec poseAdd(const PoseVec& x, const PoseVec& d)
	{
		TQuat qx(x(3),x(4),x(5),x(6));
		TQuat qd(d(3),d(4),d(5),d(6));
		TQuat qres=qx*qd;
		PoseVec ret;
		ret.Tsegment<3>(0)=x.Tsegment<3>(0)+rotatePt(qx,d.Tsegment<3>(0));
		ret(3)=qres.w();	ret(4)=qres.x();	ret(5)=qres.y();	ret(6)=qres.z();

		// gps
		ret.Tsegment<3>(7)=x.Tsegment<3>(7)+d.Tsegment<3>(7);
		return ret;
	}
private:
	typedef Quaternion<Scalar> TQuat;
	typedef Matrix<Scalar,3,1> TVec3;

	static TQuat scaledAxisToQuat(const TVec3& axis)
	{
		Scalar angle=sqrt(axis(0)*axis(0)+axis(1)*axis(1)+axis(2)*axis(2)); // axis.norm() // axis.stableNorm()

		if(angle < Scalar(1e-10))
		{
			Scalar k1(1);
			Scalar k2(0.5*axis.x());
			Scalar k3(0.5*axis.y());
			Scalar k4(0.5*axis.z());
			Scalar norma=sqrt(k1*k1+k2*k2+k3*k3+k4*k4);
			return TQuat(k1/norma,k2/norma,k3/norma,k4/norma);
			//return TQuat(Scalar(1),Scalar(0),Scalar(0),Scalar(0));
		}

		TQuat qd;
		qd.w() = cos(0.5*angle);
		qd.vec()=axis*sin(0.5*angle)/angle;
		return qd;
	}

	static TVec3 quatToScaledAxis(const TQuat& q)
	{
		Scalar nv = sqrt(q.vec()(0)*q.vec()(0) + q.vec()(1)*q.vec()(1) + q.vec()(2)*q.vec()(2));
		if( nv < Scalar(1e-12))
		{
			nv = Scalar(1e-12);
			return 2.0*q.vec();
		}
		//nv+=1e-12;

		// BUGFIX 2009-11-30: q and -q are represent the same rotation
		//                    and must lead to the same result!
		// Note that singularity for w==0 is not dramatic, as
		// atan(+/-inf) = +/- pi/2 (in this case both solutions are acceptable)
		Scalar s = 2*std::atan(nv/(q.w()))/nv; // != 2*atan2(nv, w)/nv;
		//2.0*acos(std::min(1.0,q.w()))/nv;//

		return q.vec()*s;
	}

	// for poseMinus...
	static TVec3 rotatePt(const TQuat& q, const TVec3& v)
	{
		Scalar x=q.x();
		Scalar y=q.y();
		Scalar z=q.z();
		Scalar w=q.w();

		Scalar
			//ww=w*w,
			wx=w*x, xx=x*x,
			wy=w*y, xy=x*y, yy=y*y,
			wz=w*z, xz=x*z, yz=y*z, zz=z*z;

		Scalar X= v(0), Y=v(1), Z=v(2);
		TVec3 ret;
		ret(0) = (1.0+(-2.0*zz)+(-2.0*yy))*X + 2*(xy-wz)*Y + 2*(wy+xz)*Z;
		ret(1) = 2*(xy+wz)*X + (1.0+(-2*zz)+(-2*xx))*Y + 2*(yz-wx)*Z;
		ret(2) = 2*(xz-wy)*X + 2*(yz+wx)*Y + (1.0+(-2.0*yy)+(-2.0*xx))*Z;
		return ret;
	}
#undef Tsegment
};

typedef PoseManifold3DGPS<double> PoseManifold3DGPSd;

namespace
{
//void poseToRt(const PoseManifold3DGPSd::PoseVec& pose, Matrix3d& R, Vector3d& t)
//{
//	Quaterniond quat(pose(3),pose(4),pose(5),pose(6));

//	Transform<double,3,Affine> trans;
//	trans=quat;

//	t=pose.segment<3>(0);
//	R=trans.matrix().block<3,3>(0,0);
//}

//PoseManifold3DGPSd::PoseVec RtToPose(const Matrix3d& R, const Vector3d& t)
//{
//	Quaterniond q(R);
//	PoseManifold3DGPSd::PoseVec p;
//	p.segment<3>(0)=t;
//	p(3)=q.w(); p(4)=q.x(); p(5)=q.y(); p(6)=q.z();
//	return p;
//}
}

#endif
