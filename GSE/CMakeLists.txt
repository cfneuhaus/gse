configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

#add_library(GSE SHARED foo.c foo.h config.h.in)

set (HEADERS
./include/GSE/RecBayes/ParticleFilter.h
./include/GSE/RecBayes/SqrtUKFManifold.h
./include/GSE/RecBayes/ParticleDistribution.h
./include/GSE/RecBayes/Unscented.h
./include/GSE/RecBayes/UKF.h
./include/GSE/Gaussian.h
./include/GSE/RecBayes.h
./include/GSE/LS.h
./include/GSE/GSEDefs.h
./include/GSE/PoseManifold3DGPS.h
./include/GSE/DiracDistribution.h
./include/GSE/Distributions.h
./include/GSE/SqrtGaussian.h
./include/GSE/PoseManifold2D.h
./include/GSE/LS/ConstraintTraits.h
./include/GSE/LS/GenericAbsConstraint.h
./include/GSE/LS/WrappedConstraint.h
./include/GSE/LS/NumDiff.h
./include/GSE/LS/NonlinearEstimator.h
./include/GSE/LS/LSProblem.h
./include/GSE/LS/ConstraintRef.h
./include/GSE/LS/RandomVariable.h
./include/GSE/eigen.h
./include/GSE/ManifoldUtil.h
./include/GSE/ThreadPool.h
./include/GSE/PoseManifold3D.h
)

ADD_DEFINITIONS(-DGSE_NO_CHOLMOD)


#set_target_properties(GSE PROPERTIES PUBLIC_HEADER "${HEADERS};${CMAKE_CURRENT_BINARY_DIR}/config.h")

#install(TARGETS GSE
  # IMPORTANT: Add the foo library to the "export-set"
#  EXPORT GSETargets
#  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
#  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
#  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/GSE"
#    COMPONENT dev)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "${INSTALL_INCLUDE_DIR}/GSE" FILES_MATCHING REGEX "^.*\\.h$" PATTERN "build" EXCLUDE  )
