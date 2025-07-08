add_library(project_dependencies INTERFACE)

if(APPLE)
    find_package(OpenMP REQUIRED)
    find_package(Eigen3 3.4 REQUIRED)

    set(PACKAGES
            symengine
            gmp
            mpfr
            libmpc
            flint
    )

    foreach(PKG ${PACKAGES})
        set(PKG_ROOT /opt/homebrew/opt/${PKG})
        target_include_directories(project_dependencies
                INTERFACE ${PKG_ROOT}/include
        )
        target_link_directories(project_dependencies
                INTERFACE ${PKG_ROOT}/lib
        )
        if(PKG STREQUAL "libmpc")
            set(PKG_LINK_NAME mpc)
        else()
            set(PKG_LINK_NAME ${PKG})
        endif()
        target_link_libraries(project_dependencies
                INTERFACE ${PKG_LINK_NAME}
        )
    endforeach()

    target_link_libraries(project_dependencies
            INTERFACE
            OpenMP::OpenMP_CXX
            Eigen3::Eigen
    )

elseif(WIN32)
    find_package(SymEngine CONFIG REQUIRED)
    find_package(GMP       CONFIG REQUIRED)
    find_package(MPFR      CONFIG REQUIRED)
    find_package(MPC       CONFIG REQUIRED)
    find_package(FLINT     CONFIG REQUIRED)
    find_package(OpenMP    REQUIRED)
    find_package(Eigen3    CONFIG REQUIRED)

    target_link_libraries(project_dependencies
            INTERFACE
            symengine::symengine
            GMP::GMP
            MPFR::MPFR
            MPC::MPC
            flint::flint
            OpenMP::OpenMP_CXX
            Eigen3::Eigen
    )

else()
    message(FATAL_ERROR "Unsupported platform: only MacOS or Windows are configured")
endif()