configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

#add_library(GSE SHARED foo.c foo.h config.h.in)

set (HEADERS
./RecBayes/ParticleFilter.h
./RecBayes/SqrtUKFManifold.h
./RecBayes/ParticleDistribution.h
./RecBayes/Unscented.h
./RecBayes/UKF.h
./Gaussian.h
./RecBayes.h
./LS.h
./GSEDefs.h
./DiracDistribution.h
./Distributions.h
./SqrtGaussian.h
./PoseManifold2D.h
./LS/ConstraintTraits.h
./LS/GenericAbsConstraint.h
./LS/WrappedConstraint.h
./LS/NumDiff.h
./LS/NonlinearEstimator.h
./LS/LSProblem.h
./LS/ConstraintRef.h
./LS/RandomVariable.h
./eigen.h
./ManifoldUtil.h
./ThreadPool.h
./PoseManifold3D.h
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
