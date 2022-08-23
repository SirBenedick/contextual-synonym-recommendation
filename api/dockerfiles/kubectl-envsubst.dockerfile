FROM bitnami/kubectl:latest

USER 0

RUN install_packages gettext-base

USER 1001
