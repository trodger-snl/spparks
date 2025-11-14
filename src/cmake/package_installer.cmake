# Package installer script
include("/Users/Tron/spparks/src/cmake/PackageManager.cmake")

if(SPPARKS_INSTALL_ALL_PACKAGES)
    install_all_packages()
elseif(SPPARKS_UNINSTALL_ALL_PACKAGES)
    uninstall_all_packages()
endif()
